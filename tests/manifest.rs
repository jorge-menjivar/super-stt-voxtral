// SPDX-License-Identifier: GPL-3.0-only
//! The manifest against the release workflow's names.

/// Each `[[assets.subprocess]]` block's `key = "value"` and `key = number`
/// lines, read without a TOML parser: the blocks are flat.
fn asset_blocks(manifest: &str) -> Vec<Vec<(String, String)>> {
    let mut blocks = Vec::new();
    let mut current: Option<Vec<(String, String)>> = None;
    for line in manifest.lines().map(str::trim) {
        if line.starts_with('[') {
            blocks.extend(current.take());
            if line == "[[assets.subprocess]]" {
                current = Some(Vec::new());
            }
            continue;
        }
        if let (Some(block), Some((key, value))) = (current.as_mut(), line.split_once('=')) {
            block.push((
                key.trim().to_string(),
                value.trim().trim_matches('"').to_string(),
            ));
        }
    }
    blocks.extend(current);
    blocks
}

/// Every asset is the tarball the release workflow builds for its target and
/// accelerator, `voxtral-<target>-<suffix>.tar.gz`, where the suffix is the
/// accelerator and, for CUDA, its major version. A file named anything else is
/// never attached to a release, and the indexer refuses a release missing a
/// declared asset, so the whole release would never be listed.
#[test]
fn every_asset_is_named_for_its_build() {
    let blocks = asset_blocks(include_str!("../backend.toml"));
    assert!(
        !blocks.is_empty(),
        "no [[assets.subprocess]] in backend.toml"
    );
    for block in blocks {
        let get = |key: &str| {
            block
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.as_str())
        };
        let target = get("target").expect("an asset without a target");
        let accel = get("accel").expect("an asset without an accel");
        let suffix = match accel {
            "cuda" => format!(
                "cuda{}",
                get("cuda_major").expect("a CUDA asset without cuda_major")
            ),
            other => other.to_string(),
        };
        assert_eq!(
            get("file"),
            Some(format!("voxtral-{target}-{suffix}.tar.gz").as_str()),
            "the {target} {suffix} asset"
        );
    }
}
