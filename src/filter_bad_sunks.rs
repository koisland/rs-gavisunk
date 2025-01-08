use polars::prelude::*;

pub fn filter_bad_sunks(df_sunks: &DataFrame) -> eyre::Result<DataFrame> {
    let df = df_sunks
        .select(["ctg", "group"])?
        .lazy()
        .with_column((col("ctg") + lit(":") + col("group").cast(DataType::String)).alias("id"))
        .group_by(["id"])
        // Get count of ctg+group
        .agg([col("ctg").len().alias("count")])
        // Calculate mode. Based on distribution of kmers. Dependent on sequencing technology.
        // Histogram of ONT kmer counts.
        // * x is kmer count
        // * y is the number of times x kmer count occurs.
        // * Left skewed due to error rate. We filter
        // \
        //  \     *
        //  |    /-\
        //  \___/   \_ /
        // 1 2 3 4 5 6 7
        .filter(col("count").gt(lit(2)))
        .with_column(col("count").mode().first().alias("mean_count"));

    // dbg!("{}", df.clone().collect()?);

    // Filter SUNKs with count less than 2 or are 4 root mean square/stdev above the mean.
    // https://mathworld.wolfram.com/Root-Mean-Square.html
    Ok(df
        .filter(
            col("count")
                .gt(col("mean_count") + col("mean_count").sqrt() * lit(4))
                .or(col("count").lt(2)),
        )
        .select([col("id"), col("count")])
        .collect()?)
}
