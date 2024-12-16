use std::fs::File;

use get_kmers::get_sunk_positions;
use io::Fasta;
use polars::prelude::*;

mod io;
mod get_kmers;

fn main() -> eyre::Result<()> {
    let kmer_size = 20;
    let fh = Fasta::new("test/input/all.fa")?;
    let mut df_sunks = get_sunk_positions(fh, kmer_size, true)?;
    let mut file = File::create("sunks.tsv").expect("could not create file");
    CsvWriter::new(&mut file)
        .include_header(true)
        .with_separator(b'\t')
        .finish(&mut df_sunks)?;
    Ok(())
}
