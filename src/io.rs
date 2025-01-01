use core::str;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

use coitrees::{COITree, Interval, IntervalTree};
use eyre::Context;
use itertools::Itertools;
use noodles::{
    bgzf::{self, IndexedReader},
    fasta::{self},
};
use polars::prelude::*;

pub type RegionIntervals<T> = HashMap<String, Vec<Interval<T>>>;
pub type RegionIntervalTrees<T> = HashMap<String, COITree<T, usize>>;

/// Read an input bedfile and convert it to a [`COITree`].
///
/// # Arguments
/// * `bed`: Bedfile path.
/// * `intervals_fn`: Function applied to `(start, stop, other_cols)` to convert into an [`Interval`].
///
/// # Examples
/// BED3 record.
/// ```
/// let records = read_bed(
///     "test.bed",
///     |start: i32, stop: i32, other_cols: &str| Interval::new(start, stop, None)
/// )
/// ```
/// BED4 record
/// ```
/// let records = read_bed(
///     "test.bed",
///     |start: i32, stop: i32, other_cols: &str| Interval::new(start, stop, Some(other_cols.to_owned()))
/// )
/// ```
pub fn read_bed<T: Clone>(
    bed: Option<impl AsRef<Path>>,
    intervals_fn: impl Fn(i32, i32, &str) -> Interval<T>,
) -> eyre::Result<Option<RegionIntervalTrees<T>>> {
    let mut intervals: RegionIntervals<T> = HashMap::new();
    let mut trees: RegionIntervalTrees<T> = HashMap::new();

    let Some(bed) = bed else {
        return Ok(None);
    };
    let bed_fh = File::open(bed)?;
    let bed_reader = BufReader::new(bed_fh);

    for line in bed_reader.lines() {
        let line = line?;
        let (name, start, stop, other_cols) =
            if let Some((name, start, stop, other_cols)) = line.splitn(4, '\t').collect_tuple() {
                (name, start, stop, other_cols)
            } else if let Some((name, start, stop)) = line.splitn(3, '\t').collect_tuple() {
                (name, start, stop, "")
            } else {
                log::error!("Invalid line: {line}");
                continue;
            };
        let (first, last) = (start.parse::<i32>()?, stop.parse::<i32>()?);

        intervals
            .entry(name.to_owned())
            .and_modify(|intervals| intervals.push(intervals_fn(first, last, other_cols)))
            .or_insert_with(|| vec![intervals_fn(first, last, other_cols)]);
    }
    for (roi, intervals) in intervals.into_iter() {
        trees.entry(roi).or_insert(COITree::new(&intervals));
    }
    Ok(Some(trees))
}

pub enum FastaReader {
    Bgzip(fasta::io::Reader<IndexedReader<File>>),
    Standard(fasta::io::Reader<BufReader<File>>),
}

pub struct Fasta {
    pub fname: PathBuf,
    reader: FastaReader,
    index: fasta::fai::Index,
}

impl Fasta {
    pub fn new(infile: impl AsRef<Path>) -> eyre::Result<Self> {
        let fname = infile.as_ref().to_owned();
        let (index, gzi) = Self::get_faidx(&infile)?;
        let fh = Self::read_fa(&infile, gzi.as_ref())?;
        Ok(Self {
            fname,
            reader: fh,
            index,
        })
    }

    pub fn lengths(&self) -> Vec<(String, u64)> {
        self.index
            .as_ref()
            .iter()
            .map(|rec| {
                (
                    String::from_utf8(rec.name().to_vec()).unwrap(),
                    rec.length(),
                )
            })
            .collect()
    }

    fn get_faidx(
        fa: &impl AsRef<Path>,
    ) -> eyre::Result<(fasta::fai::Index, Option<bgzf::gzi::Index>)> {
        // https://www.ginkgobioworks.com/2023/03/17/even-more-rapid-retrieval-from-very-large-files-with-rust/
        let fa_path = fa.as_ref().canonicalize()?;
        let is_bgzipped = fa_path.extension().and_then(|e| e.to_str()) == Some("gz");
        let fai_fname = fa_path.with_extension(if is_bgzipped { "gz.fai" } else { "fa.fai" });
        let fai = fasta::fai::read(fai_fname);
        if is_bgzipped {
            let index_reader = bgzf::indexed_reader::Builder::default()
                .build_from_path(fa)
                .with_context(|| format!("Failed to read gzi for {fa_path:?}"))?;
            let gzi = index_reader.index().clone();

            if let Ok(fai) = fai {
                log::debug!("Existing fai index found for {fa_path:?}");
                return Ok((fai, Some(gzi)));
            }
            log::debug!("No existing faidx for {fa_path:?}. Generating...");
            let mut records = Vec::new();
            let mut indexer = fasta::io::Indexer::new(index_reader);
            while let Some(record) = indexer.index_record()? {
                records.push(record);
            }

            Ok((fasta::fai::Index::from(records), Some(gzi)))
        } else {
            if let Ok(fai) = fai {
                return Ok((fai, None));
            }
            log::debug!("No existing faidx for {fa_path:?}. Generating...");
            Ok((fasta::index(fa)?, None))
        }
    }

    pub fn fetch(&mut self, ctg_name: &str, start: u32, stop: u32) -> eyre::Result<fasta::Record> {
        let start_pos = noodles::core::Position::new(start.clamp(1, u32::MAX) as usize).unwrap();
        let stop_pos = noodles::core::Position::new(stop.clamp(1, u32::MAX) as usize).unwrap();
        let region = noodles::core::Region::new(ctg_name, start_pos..=stop_pos);
        match &mut self.reader {
            FastaReader::Bgzip(reader) => Ok(reader.query(&self.index, &region)?),
            FastaReader::Standard(reader) => Ok(reader.query(&self.index, &region)?),
        }
    }

    fn read_fa(
        fa: &impl AsRef<Path>,
        fa_gzi: Option<&bgzf::gzi::Index>,
    ) -> eyre::Result<FastaReader> {
        let fa_file = std::fs::File::open(fa);
        if let Some(fa_gzi) = fa_gzi {
            Ok(FastaReader::Bgzip(
                fa_file
                    .map(|file| bgzf::IndexedReader::new(file, fa_gzi.to_vec()))
                    .map(fasta::io::Reader::new)?,
            ))
        } else {
            Ok(FastaReader::Standard(
                fa_file
                    .map(std::io::BufReader::new)
                    .map(fasta::io::Reader::new)?,
            ))
        }
    }
}

pub fn write_tsv(df: &mut DataFrame, path: impl AsRef<Path>) -> eyre::Result<()> {
    let mut file = File::create(path)?;
    CsvWriter::new(&mut file)
        .include_header(true)
        .with_separator(b'\t')
        .finish(df)?;
    Ok(())
}

pub fn load_tsv(path: impl AsRef<Path>) -> eyre::Result<DataFrame> {
    Ok(CsvReadOptions::default()
        .with_has_header(true)
        .with_parse_options(CsvParseOptions::default().with_separator(b'\t'))
        .try_into_reader_with_file_path(Some(PathBuf::from(path.as_ref())))?
        .finish()?)
}

/// Loads the given file if it exists. If not, then redoes function call.
///
/// # Arguments
/// * `path`
///     * File path to TSV with header.
/// * `fn_call`
///     * Expression that generates a [`DataFrame`].
///     * This will be written to `path`.
/// # Returns
/// * [`DataFrame`]
macro_rules! load_or_redo_df {
    ($path:ident, $fn_call:expr) => {
        if $path.exists() {
            log::info!("Loading existing file: {:?}", $path);
            load_tsv($path)?
        } else {
            let mut df = $fn_call;
            write_tsv(&mut df, $path)?;
            df
        }
    };
}
