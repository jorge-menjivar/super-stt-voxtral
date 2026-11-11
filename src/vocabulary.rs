// SPDX-License-Identifier: GPL-3.0-only
//! The user's vocabulary, applied to a finished transcript.
//!
//! The manifest declares `[capabilities] context = true`, so the daemon sends
//! the dictation context in force with every `/v1` request. Its vocabulary
//! arrives as an `x-voice-vocabulary` header, or `x-stt-vocabulary` from a
//! Super STT daemon: a JSON array of terms spelled the way the user wants them.
//!
//! Voxtral has nowhere to read it. Its transcription prompt is the audio and a
//! `lang:` tag, and Mistral's own encoding of a transcription request takes
//! nothing else. Its chat mode does take text with the audio, but given the
//! vocabulary there it answered spoken questions, refused spoken commands,
//! wrote out the term list instead of the transcript, and looped. So the
//! vocabulary is applied to what transcription mode wrote, in two passes:
//!
//! 1. **Exact.** A run of words whose letters and digits, lowercased and run
//!    together, are a term's is written the way the vocabulary spells it:
//!    `cube CL` becomes `CubeCL`, `QWEN` becomes `Qwen`.
//! 2. **Close.** A word, or a run of as many words as the term has, give or
//!    take one, that misspells a term in its vowels alone: `Kakaro` becomes
//!    `Kokoro`, `Super Libra` becomes `Super Libre`. Letter by letter it is as
//!    long as the term, starts with the same letter, and differs only where
//!    both have a vowel, in at most two letters out of every five. A term
//!    shorter than [`SHORTEST_CLOSE`] letters is matched exactly or not at
//!    all: a short word has too many real neighbours, `Paris` and `parish`.
//!
//! A misspelling that changes a consonant, `Mendevar` for `Menjivar`, is left
//! alone: telling it from a different word would take knowing how both sound.
//! The context's other half, its prompt, is sent too and not read.

use std::ops::Range;

use axum::http::HeaderMap;

/// The header the vocabulary comes in, then its Super STT spelling.
const HEADERS: [&str; 2] = ["x-voice-vocabulary", "x-stt-vocabulary"];

/// The shortest key a term is matched by at all. A one-letter term would
/// match every "a" and "I".
const SHORTEST_KEY: usize = 2;

/// The shortest key a term is matched to a close misspelling by.
const SHORTEST_CLOSE: usize = 6;

/// One term: as the user spells it, and the key it is matched by.
struct Term {
    spelling: String,
    key: Vec<char>,
    words: usize,
}

/// A word of a text, by its byte offsets. `core` is the word without the
/// punctuation around it or a possessive `'s`, which a replacement keeps.
struct Token {
    core: Range<usize>,
    key: Vec<char>,
}

/// The terms of the dictation context in force. Empty when there is none.
pub struct Vocabulary {
    /// Longest key first, so a term inside another ("Super" in "Super Voice")
    /// never takes the longer one's place.
    terms: Vec<Term>,
}

impl Vocabulary {
    /// The vocabulary a request carries. Absent or unreadable is empty: a
    /// malformed header is the daemon's bug, and the audio is still
    /// transcribed.
    pub fn from_headers(headers: &HeaderMap) -> Self {
        let Some((name, value)) = HEADERS
            .iter()
            .find_map(|&name| headers.get(name).map(|value| (name, value)))
        else {
            return Self::new(&[] as &[&str]);
        };
        let terms = value
            .to_str()
            .map_err(|e| e.to_string())
            .and_then(|raw| serde_json::from_str::<Vec<String>>(raw).map_err(|e| e.to_string()));
        match terms {
            Ok(terms) => Self::new(&terms),
            Err(e) => {
                log::warn!("{name} is not a JSON array of strings ({e}); ignoring it");
                Self::new(&[] as &[&str])
            }
        }
    }

    fn new<S: AsRef<str>>(terms: &[S]) -> Self {
        let mut terms: Vec<Term> = terms
            .iter()
            .filter_map(|term| {
                let spelling = term.as_ref().trim().to_string();
                let key = key(&spelling);
                (key.len() >= SHORTEST_KEY).then(|| Term {
                    words: spelling.split_whitespace().count(),
                    spelling,
                    key,
                })
            })
            .collect();
        terms.sort_by_key(|term| std::cmp::Reverse(term.key.len()));
        Self { terms }
    }

    /// `text` with each term written the way the vocabulary spells it.
    pub fn spell(&self, text: &str) -> String {
        if self.terms.is_empty() {
            return text.to_string();
        }
        let tokens = tokens(text);
        let mut taken = vec![false; tokens.len()];
        let mut edits: Vec<(Range<usize>, usize)> = Vec::new();

        // Exact first, everywhere, so that a close match never takes words an
        // exact one wanted.
        let mut start = 0;
        while start < tokens.len() {
            let hit = self
                .terms
                .iter()
                .enumerate()
                .find_map(|(term, t)| spanning(&tokens, start, &t.key).map(|end| (term, end)));
            match hit {
                Some((term, end)) => {
                    taken[start..end].fill(true);
                    edits.push((start..end, term));
                    start = end;
                }
                None => start += 1,
            }
        }
        let mut start = 0;
        while start < tokens.len() {
            match self.closest(&tokens, &taken, start) {
                Some((term, end)) => {
                    taken[start..end].fill(true);
                    edits.push((start..end, term));
                    start = end;
                }
                None => start += 1,
            }
        }

        edits.sort_by_key(|(span, _)| span.start);
        let mut out = String::with_capacity(text.len());
        let mut at = 0;
        for (span, term) in edits {
            let core = tokens[span.start].core.start..tokens[span.end - 1].core.end;
            let spelling = &self.terms[term].spelling;
            if &text[core.clone()] != spelling {
                log::debug!("spelling {:?} as {spelling:?}", &text[core.clone()]);
            }
            out.push_str(&text[at..core.start]);
            out.push_str(spelling);
            at = core.end;
        }
        out.push_str(&text[at..]);
        out
    }

    /// The term a run of untaken words from `start` misspells in its vowels,
    /// and the run's end: the one with the fewest vowels swapped, if any.
    fn closest(&self, tokens: &[Token], taken: &[bool], start: usize) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize, usize)> = None;
        for (index, term) in self.terms.iter().enumerate() {
            if term.key.len() < SHORTEST_CLOSE {
                continue;
            }
            let mut joined: Vec<char> = Vec::new();
            // As many words as the term has, give or take one: a model splits
            // and joins words it does not know.
            for end in start..tokens.len().min(start + term.words + 1) {
                if taken[end] || tokens[end].key.is_empty() {
                    break;
                }
                joined.extend(&tokens[end].key);
                if let Some(swaps) = vowel_swaps(&joined, &term.key)
                    && best.is_none_or(|(_, _, fewest)| swaps < fewest)
                {
                    best = Some((index, end + 1, swaps));
                }
            }
        }
        best.map(|(term, end, _)| (term, end))
    }
}

fn is_vowel(c: char) -> bool {
    matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'y')
}

/// How many vowels `heard` has where `term` has another, if that is all that
/// tells them apart and there are few enough: the same length, the same first
/// letter, and at most two swaps in every five letters.
fn vowel_swaps(heard: &[char], term: &[char]) -> Option<usize> {
    if heard.len() != term.len() || heard.first() != term.first() || heard == term {
        return None;
    }
    let mut swaps = 0;
    for (&h, &t) in heard.iter().zip(term) {
        if h != t {
            if !(is_vowel(h) && is_vowel(t)) {
                return None;
            }
            swaps += 1;
        }
    }
    (swaps * 5 <= term.len() * 2).then_some(swaps)
}

/// The end of the span from `start` whose keys, run together, are `key`.
fn spanning(tokens: &[Token], start: usize, key: &[char]) -> Option<usize> {
    let mut joined: Vec<char> = Vec::new();
    for (end, token) in tokens.iter().enumerate().skip(start) {
        if token.key.is_empty() {
            return None;
        }
        joined.extend(&token.key);
        if joined == key {
            return Some(end + 1);
        }
        if !key.starts_with(&joined) {
            return None;
        }
    }
    None
}

/// The words of `text`, split at whitespace.
fn tokens(text: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut start = None;
    for (at, c) in text
        .char_indices()
        .chain(std::iter::once((text.len(), ' ')))
    {
        match (start, c.is_whitespace()) {
            (None, false) => start = Some(at),
            (Some(from), true) => {
                tokens.push(token(text, from..at));
                start = None;
            }
            _ => {}
        }
    }
    tokens
}

/// One word: its core is the run from its first letter or digit to its last,
/// less a possessive `'s`.
fn token(text: &str, word: Range<usize>) -> Token {
    let w = &text[word.clone()];
    let Some(first) = w.find(char::is_alphanumeric) else {
        return Token {
            core: word.end..word.end,
            key: Vec::new(),
        };
    };
    let last = w
        .char_indices()
        .rev()
        .find(|(_, c)| c.is_alphanumeric())
        .map_or(w.len(), |(i, c)| i + c.len_utf8());
    let mut end = last;
    for suffix in ["'s", "’s"] {
        if w[first..last].len() > suffix.len() && w[first..last].ends_with(suffix) {
            end = last - suffix.len();
            break;
        }
    }
    let core = word.start + first..word.start + end;
    Token {
        key: key(&text[core.clone()]),
        core,
    }
}

/// What a term or word is matched by: its letters and digits, lowercase.
fn key(text: &str) -> Vec<char> {
    text.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spell(terms: &[&str], text: &str) -> String {
        Vocabulary::new(terms).spell(text)
    }

    #[test]
    fn spells_a_term_the_way_the_vocabulary_does() {
        assert_eq!(spell(&["CubeCL"], "ran on cube CL."), "ran on CubeCL.");
        assert_eq!(spell(&["Qwen"], "the QWEN release"), "the Qwen release");
        assert_eq!(spell(&["Burn"], "ran them on burn."), "ran them on Burn.");
        assert_eq!(spell(&["Kokoro"], "kokoro's voice"), "Kokoro's voice");
    }

    /// What Voxtral wrote for natural speech with no vocabulary, and what the
    /// vocabulary makes of it.
    #[test]
    fn fixes_what_voxtral_misspelled() {
        assert_eq!(
            spell(
                &["Super Libre", "Super Voice", "Kokoro"],
                "The Super Libra team shipped Super Voice and Kakaro today."
            ),
            "The Super Libre team shipped Super Voice and Kokoro today."
        );
    }

    #[test]
    fn a_consonant_changed_is_left_alone() {
        let text = "Please send the draft to Jorge Mendevar.";
        assert_eq!(spell(&["Jorge Menjivar"], text), text);
        assert_eq!(spell(&["Kokoro"], "a Kakarot figure"), "a Kakarot figure");
    }

    /// Real words a close match must not take for a term.
    #[test]
    fn leaves_unlike_words_alone() {
        for (terms, text) in [
            (&["Super Voice"][..], "That was super nice of you."),
            (&["Paris"], "The parish meeting is at noon."),
            (
                &["Super Libre", "Libre"],
                "I went to the library and the super market.",
            ),
            (&["Kokoro"], "I bought cocoa at the coco market."),
            (&["Menjivar"], "Talk to the manager about it."),
            (&["Kubernetes"], "The governors met on Tuesday."),
            (&["Mariana"], "We walked along the marina."),
            (&["Lorena"], "Lauren is coming tomorrow."),
            (&["Kokoro"], "We flew to Kokomo, Indiana."),
            (&["Sienna"], "The scene was set."),
            (&["Daniela"], "Daniel and Danielle are here."),
            (&["Kenia"], "We traveled to Kenya."),
            (&["Bryan"], "Brian called."),
            (&["Burn"], "The old barn."),
        ] {
            assert_eq!(spell(terms, text), text, "{terms:?}");
        }
    }

    #[test]
    fn vowel_swaps_are_counted_letter_by_letter() {
        let k = |s: &str| key(s);
        assert_eq!(vowel_swaps(&k("kakaro"), &k("kokoro")), Some(2));
        assert_eq!(vowel_swaps(&k("superlibra"), &k("superlibre")), Some(1));
        // Three in six letters is more than two in five.
        assert_eq!(vowel_swaps(&k("kikara"), &k("kokoro")), None);
        // Not the same first letter.
        assert_eq!(vowel_swaps(&k("akoro"), &k("ekoro")), None);
        // A consonant differs.
        assert_eq!(vowel_swaps(&k("kokomo"), &k("kokoro")), None);
    }

    #[test]
    fn the_longer_term_wins() {
        assert_eq!(
            spell(&["Super", "Super Voice"], "super voice"),
            "Super Voice"
        );
    }

    #[test]
    fn a_one_letter_term_is_ignored() {
        assert_eq!(spell(&["A"], "a cat"), "a cat");
    }

    #[test]
    fn no_vocabulary_changes_nothing() {
        let text = "The Super Libra team.";
        assert_eq!(spell(&[], text), text);
    }

    fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for &(name, value) in pairs {
            map.insert(name, value.parse().unwrap());
        }
        map
    }

    #[test]
    fn reads_the_header_and_its_super_stt_name() {
        let text = "ran on cube CL";
        let voice = headers(&[("x-voice-vocabulary", r#"["CubeCL"]"#)]);
        assert_eq!(
            Vocabulary::from_headers(&voice).spell(text),
            "ran on CubeCL"
        );
        let stt = headers(&[("x-stt-vocabulary", r#"["CubeCL"]"#)]);
        assert_eq!(Vocabulary::from_headers(&stt).spell(text), "ran on CubeCL");
        // A Super Voice daemon sends both, alike; the new name is read.
        let both = headers(&[
            ("x-voice-vocabulary", r#"["CubeCL"]"#),
            ("x-stt-vocabulary", r#"["cubecl"]"#),
        ]);
        assert_eq!(Vocabulary::from_headers(&both).spell(text), "ran on CubeCL");
    }

    #[test]
    fn reads_a_term_escaped_to_ascii() {
        let map = headers(&[("x-voice-vocabulary", r#"["Men\u00edvar"]"#)]);
        assert_eq!(
            Vocabulary::from_headers(&map).spell("menívar said"),
            "Menívar said"
        );
    }

    #[test]
    fn an_unreadable_header_is_no_vocabulary() {
        let text = "ran on cube CL";
        for raw in ["not json", r#""CubeCL""#, r"[1, 2]"] {
            let map = headers(&[("x-voice-vocabulary", raw)]);
            assert_eq!(Vocabulary::from_headers(&map).spell(text), text, "{raw}");
        }
        assert_eq!(
            Vocabulary::from_headers(&HeaderMap::new()).spell(text),
            text
        );
    }
}
