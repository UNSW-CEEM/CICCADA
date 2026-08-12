"""Legacy preliminary data inspection retained for reference."""

import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from funcs import loadCleanedSiteData

site_details = pl.read_csv("Nov2022/ebm_1_20221112_20221119_site_details.csv")
circuit_details = pl.read_csv("Nov2022/ebm_1_20221112_20221119_circuit_details.csv")

data_df = loadCleanedSiteData().collect()

# see total number of sites
observed_sites = (
    data_df.join(circuit_details.select(["c_id", "site_id"]), on="c_id")
    .select("site_id")
    .drop_nulls()
    .unique()
)  # join on circuit id and slect c_id ann site_id only
numSites = observed_sites["site_id"].n_unique()

# see if the number matches in meta data
numSitesMeta = site_details["site_id"].n_unique()
numSitesMeta2 = circuit_details["site_id"].n_unique()

# Sites in metadata with no circuits
sites_orphan_in_metadata = site_details.join(
    circuit_details.select("site_id").unique(), on="site_id", how="anti"
).select("site_id")
if sites_orphan_in_metadata.height > 1:
    print("Sites in data without any circuits")

if numSites == numSitesMeta:
    pass
else:
    # Sites that appear in data (via circuits)
    sites_with_data = (
        data_df.select("c_id")
        .unique()
        .join(
            circuit_details.select(["c_id", "site_id"]).unique(), on="c_id", how="inner"
        )
        .select("site_id")
        .unique()
    )
    # All sites that have at least one circuit
    sites_with_circuits = circuit_details.select("site_id").unique()
    # Sites that have circuits but no data in the loaded period
    sites_with_circuits_but_no_data = sites_with_circuits.join(
        sites_with_data, on="site_id", how="anti"
    )
    # sites_orphan_in_metadata is empty but added for safety
    if (
        sites_with_circuits_but_no_data.height
        + sites_with_data.height
        + sites_orphan_in_metadata.height
        == sites_with_circuits.height
    ):
        print("Number of sites without data: ", sites_with_circuits_but_no_data.height)
    else:
        raise ValueError("Num Sites not Consistent")

# see number of sites with solar and without
pv_sites = (
    circuit_details.filter(pl.col("con_type") == "pv_site_net")
    .select("site_id")
    .unique()
)
non_pv_sites = (
    circuit_details.select("site_id").unique().join(pv_sites, on="site_id", how="anti")
)
totalSites = pv_sites.height + non_pv_sites.height

# PV presence by inverter_manufacturer
# doing this to avoid one site having multiple PVs
all_sites = site_details.select("site_id").drop_nulls().unique()
sites_with_inverter = (
    site_details.filter(
        pl.col("inverter_manufacturer").is_not_null()
        & (pl.col("inverter_manufacturer").str.strip_chars() != "")
    )
    .select("site_id")
    .drop_nulls()
    .unique()
)
# Sites without inverter = all_sites \ sites_with_inverter
sites_without_inverter = all_sites.join(sites_with_inverter, on="site_id", how="anti")
totalSitesFromInverter = sites_with_inverter.height + sites_without_inverter.height

# it should add up to the total number of sites
# add PV + non-PV sites to see if they match
if numSitesMeta != totalSites or numSitesMeta != totalSitesFromInverter:
    raise ValueError("Num Sites not Consistent")

# see number of sites that have more than one PV circuit
sites_multi_pv = (
    circuit_details.filter(pl.col("con_type") == "pv_site_net")
    .group_by("site_id")
    .agg(pl.n_unique("c_id").alias("pv_count"))
    .filter(pl.col("pv_count") > 1)
)
print("Number of sites with >1 PV circuit:", sites_multi_pv.height)

# Circuit IDs present in data but missing in circuit metadata
orphan_cids_in_data = (
    data_df.select("c_id")
    .unique()
    .join(circuit_details.select("c_id").unique(), on="c_id", how="anti")
)
if orphan_cids_in_data.height > 1:
    print("Num {} of circuits in data but missing in circuit metadata").format(
        orphan_cids_in_data.height
    )

# Circuits in metadata with no sites
# other way you could just do is see if they have sites misisng next to them in metadata
circuits_orphan_in_metadata = circuit_details.join(
    site_details.select("site_id").unique(), on="site_id", how="anti"
).select(["c_id", "site_id"])
if circuits_orphan_in_metadata.height:
    print("Num {} of circuits in metadata but no sites")


# see if each circuit id is only avaiable to one site
def validate_unique_circuit_to_site(circuit_details: pl.DataFrame) -> pl.DataFrame:
    cid_site_card = (
        circuit_details.group_by("c_id")
        .agg(
            pl.col("site_id").n_unique().alias("site_count"),
            pl.col("site_id").unique().alias("sites"),
        )
        .filter(pl.col("site_count") > 1)
    )
    if cid_site_card.height > 0:
        print("WARNING: Circuits mapped to multiple sites (unexpected):")
        print(cid_site_card)
    else:
        print("OK: c_id → site_id mapping is unique.")
    return cid_site_card


multi_site_cids = validate_unique_circuit_to_site(circuit_details)


# is it okay for a site to have multiple PV circuits but only one AC circuit?
# or num ac circuits < pv circuits?
# needs confirmation
# sites that have more PVs than AC circuits net

# Per-site counts
pv_counts = (
    circuit_details.filter(pl.col("con_type") == "pv_site_net")
    .group_by("site_id")
    .agg(pl.n_unique("c_id").alias("pv_count"))  # DISTINCT c_id per site
)

ac_counts = (
    circuit_details.filter(pl.col("con_type") == "ac_load_net")
    .group_by("site_id")
    .agg(pl.n_unique("c_id").alias("ac_count"))
)

# Sites where PV count > AC load count (treat missing AC as 0)
sites_more_pv = (
    pv_counts.join(ac_counts, on="site_id", how="left")
    .with_columns(pl.col("ac_count").fill_null(0))
    .filter(pl.col("pv_count") > pl.col("ac_count"))
)

print("Sites where PV circuits > ac_load_net circuits:", sites_more_pv.height)

if (
    pv_counts["pv_count"].sum()
    != circuit_details.filter(pl.col("con_type") == "pv_site_net")["c_id"].n_unique()
):
    raise ValueError("PV count needs to be same")
else:
    print("PV count matches")
# Optional: inspect
# sites_more_pv.select(["site_id", "pv_count", "ac_count"]).head()

# is it okay for a site to have multiple AC circuits but only one PV circuit?
# Yep, check sublime doc

# apart from 250ms reso and 1m reso
# the reso is given for circuits not sites
# so perhaps filter out pv sites and then check the reso?
circuitsWIth250Reso = (
    circuit_details.filter(pl.col("con_type") == "pv_site_net")
    .filter(pl.col("250ms_voltage") == True)
    .unique(subset="c_id")
)

percentageCircuit250Reso = (
    circuitsWIth250Reso.height / pv_counts["pv_count"].sum() * 100
)
print("Percentage of circuits with 250ms {}".format(percentageCircuit250Reso))


pct_250 = (
    circuit_details.filter(pl.col("con_type") == "pv_site_net")
    .with_columns(pl.col("250ms_voltage").fill_null(False))
    .select((pl.col("250ms_voltage").mean() * 100).alias("pct_250"))
).item()

# maybe also look if there are other resolutions?

# confirm data and duration of data? - for each circuit?
# needs thinking
# do it separately?
print("Analysis done")
