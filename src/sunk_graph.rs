use std::collections::HashMap;

use polars::prelude::*;

pub fn create_sunk_graph(
    df_asm_sunks: &DataFrame,
    asm_lens: &HashMap<String, u64>,
    df_read_sunks: &DataFrame,
    read_lens: &HashMap<String, u64>,
    df_bad_sunks: &DataFrame,
) -> eyre::Result<()> {
    Ok(())
}
