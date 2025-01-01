use std::path::Path;

use assign_read_ctg::assign_read_to_ctg_w_ort;
use filter_bad_sunks::filter_bad_sunks;
use get_kmers::get_sunk_positions;
use io::{load_tsv, write_tsv, Fasta};
use map_kmers::{get_good_read_sunks, map_sunks_to_reads};

mod assign_read_ctg;
mod get_kmers;
#[macro_use]
mod io;
mod filter_bad_sunks;
mod map_kmers;

fn main() -> eyre::Result<()> {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Info)
        .init()?;

    let kmer_size = 20;
    let asm_fh = Fasta::new("test/input/all.fa")?;
    let ont_fh = Fasta::new("test/input/all_ONT.fa")?;

    log::info!("Getting SUNK positions in assembly.");
    let path_sunks_asm = Path::new("asm_sunks.tsv");
    let df_asm_sunks = load_or_redo_df!(path_sunks_asm, get_sunk_positions(asm_fh, kmer_size)?);

    log::info!("Mapping assembly SUNKs to reads.");
    let path_sunks_reads = Path::new("read_sunks.tsv");
    let df_read_sunks =
        load_or_redo_df!(path_sunks_reads, map_sunks_to_reads(ont_fh, &df_asm_sunks)?);

    log::info!("Assigning reads to assembly contigs.");
    let path_best_reads_asm = Path::new("read_ctg_mapping.tsv");
    let df_best_reads_asm = load_or_redo_df!(
        path_best_reads_asm,
        assign_read_to_ctg_w_ort(&df_read_sunks, None, None)?
    );

    log::info!("Filtering read SUNKs.");
    let path_good_sunks_reads = Path::new("read_sunks_good.tsv");
    let df_good_sunks_reads = load_or_redo_df!(
        path_good_sunks_reads,
        get_good_read_sunks(&df_read_sunks, &df_best_reads_asm)?
    );

    // TODO: Filter bad sunks
    filter_bad_sunks(df_good_sunks_reads);

    // TODO: Process by contig

    log::info!("Done.");
    Ok(())
}
