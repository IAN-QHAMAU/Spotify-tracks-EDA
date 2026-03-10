from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DATA_PATH = Path("data/spotify tracks dataset.csv")
OUT_DIR = Path("visualisations/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    unnamed_cols = [col for col in df.columns if col.startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    numeric_cols = [
        "popularity",
        "duration_ms",
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "duration_ms" in df.columns:
        df["duration_min"] = df["duration_ms"] / 60000

    if "explicit" in df.columns:
        df["explicit"] = (
            df["explicit"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False})
        )

    return df


def save_fig(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=180)
    plt.close()


def visual_5_genre_fingerprint(df: pd.DataFrame, findings: list[str]) -> None:
    req = {"track_genre", "energy", "valence", "popularity"}
    if not req.issubset(df.columns):
        return

    genre_stats = (
        df.dropna(subset=list(req))
        .groupby("track_genre", as_index=False)
        .agg(
            tracks=("track_genre", "size"),
            energy=("energy", "median"),
            valence=("valence", "median"),
            popularity=("popularity", "median"),
        )
    )
    genre_stats = genre_stats[genre_stats["tracks"] >= 120].copy()
    top = genre_stats.sort_values("popularity", ascending=False).head(20)

    plt.figure(figsize=(11, 8))
    scatter = plt.scatter(
        genre_stats["energy"],
        genre_stats["valence"],
        s=np.sqrt(genre_stats["tracks"]) * 20,
        c=genre_stats["popularity"],
        cmap="viridis",
        alpha=0.7,
        edgecolors="black",
        linewidths=0.3,
    )
    plt.colorbar(scatter, label="Median Popularity")
    for _, row in top.iterrows():
        plt.text(row["energy"] + 0.005, row["valence"] + 0.005, row["track_genre"], fontsize=8)
    plt.title("Impact 5/5 — Genre Fingerprint Map\n(Energy vs Valence, bubble size = tracks)")
    plt.xlabel("Median Energy")
    plt.ylabel("Median Valence (Positivity)")
    save_fig("impact_5_genre_fingerprint.png")

    quadrant = top.assign(
        mood=np.where(
            (top["energy"] > top["energy"].median()) & (top["valence"] > top["valence"].median()),
            "High-energy positive",
            "Other",
        )
    )
    share = (quadrant["mood"] == "High-energy positive").mean() * 100
    findings.append(
        f"Impact 5/5: Genre fingerprint map shows where popularity clusters by mood. "
        f"Among top-20 genres by median popularity, {share:.1f}% sit in the high-energy/high-valence quadrant."
    )


def visual_4_hit_zone_heatmap(df: pd.DataFrame, findings: list[str]) -> None:
    req = {"danceability", "energy", "popularity"}
    if not req.issubset(df.columns):
        return

    data = df.dropna(subset=list(req)).copy()
    data = data[(data["danceability"].between(0, 1)) & (data["energy"].between(0, 1))]

    data["dance_bin"] = pd.cut(data["danceability"], bins=np.linspace(0, 1, 11), include_lowest=True)
    data["energy_bin"] = pd.cut(data["energy"], bins=np.linspace(0, 1, 11), include_lowest=True)

    heat = data.pivot_table(
        index="energy_bin",
        columns="dance_bin",
        values="popularity",
        aggfunc="median",
        observed=False,
    )

    plt.figure(figsize=(11, 8))
    sns.heatmap(heat, cmap="magma", cbar_kws={"label": "Median Popularity"})
    plt.title("Impact 4/5 — Hit Zone Heatmap\n(Median popularity by danceability & energy bins)")
    plt.xlabel("Danceability bins")
    plt.ylabel("Energy bins")
    save_fig("impact_4_hit_zone_heatmap.png")

    best_cell = heat.stack().idxmax() if not heat.empty else None
    if best_cell:
        findings.append(
            f"Impact 4/5: Highest median popularity appears around danceability {best_cell[1]} and energy {best_cell[0]}."
        )


def visual_3_artist_consistency(df: pd.DataFrame, findings: list[str]) -> None:
    req = {"artists", "popularity"}
    if not req.issubset(df.columns):
        return

    artist = (
        df.dropna(subset=list(req))
        .groupby("artists", as_index=False)
        .agg(tracks=("artists", "size"), mean_pop=("popularity", "mean"), std_pop=("popularity", "std"))
    )
    artist = artist[artist["tracks"] >= 12].dropna(subset=["std_pop"])
    artist = artist.sort_values("mean_pop", ascending=False).head(80)
    if artist.empty:
        return

    plt.figure(figsize=(11, 8))
    sns.scatterplot(
        data=artist,
        x="std_pop",
        y="mean_pop",
        size="tracks",
        sizes=(40, 450),
        hue="tracks",
        palette="crest",
        alpha=0.8,
        legend=False,
    )
    for _, row in artist.sort_values("mean_pop", ascending=False).head(12).iterrows():
        plt.text(row["std_pop"] + 0.3, row["mean_pop"] + 0.2, row["artists"], fontsize=7)
    plt.title("Impact 3/5 — Artist Consistency vs Popularity\n(Lower spread + high mean = reliable hit-makers)")
    plt.xlabel("Popularity Volatility (Std Dev)")
    plt.ylabel("Average Popularity")
    save_fig("impact_3_artist_consistency.png")

    star = artist.sort_values(["mean_pop", "std_pop"], ascending=[False, True]).head(1)
    if not star.empty:
        row = star.iloc[0]
        findings.append(
            f"Impact 3/5: Artist consistency map highlights '{row['artists']}' as a high-popularity, low-volatility standout "
            f"(mean={row['mean_pop']:.1f}, std={row['std_pop']:.1f})."
        )


def visual_2_tempo_loudness(df: pd.DataFrame, findings: list[str]) -> None:
    req = {"tempo", "loudness", "popularity"}
    if not req.issubset(df.columns):
        return

    data = df.dropna(subset=list(req)).copy()
    data = data[(data["tempo"].between(40, 220)) & (data["loudness"].between(-30, 2))]
    if data.empty:
        return

    sample = data.sample(min(20000, len(data)), random_state=42)
    plt.figure(figsize=(11, 8))
    hb = plt.hexbin(
        sample["tempo"],
        sample["loudness"],
        C=sample["popularity"],
        gridsize=35,
        reduce_C_function=np.mean,
        cmap="plasma",
        mincnt=15,
    )
    plt.colorbar(hb, label="Mean Popularity")
    plt.title("Impact 2/5 — Tempo-Loudness Hit Terrain\n(Hexbin color = mean popularity)")
    plt.xlabel("Tempo (BPM)")
    plt.ylabel("Loudness (dB)")
    save_fig("impact_2_tempo_loudness_hit_terrain.png")

    hot = sample.assign(
        score=(sample["popularity"] - sample["popularity"].mean())
    ).sort_values("score", ascending=False)
    top_tempo = hot["tempo"].head(1000).median()
    top_loud = hot["loudness"].head(1000).median()
    findings.append(
        f"Impact 2/5: Higher-performing tracks tend to center around tempo ~{top_tempo:.0f} BPM and loudness ~{top_loud:.1f} dB."
    )


def visual_1_duration_popularity(df: pd.DataFrame, findings: list[str]) -> None:
    req = {"duration_min", "popularity"}
    if not req.issubset(df.columns):
        return

    data = df.dropna(subset=list(req)).copy()
    data = data[data["duration_min"].between(1.0, 8.0)]
    if data.empty:
        return

    data["duration_bin"] = pd.cut(data["duration_min"], bins=np.arange(1, 8.5, 0.5), include_lowest=True)
    trend = data.groupby("duration_bin", observed=True)["popularity"].median().reset_index()
    trend["mid"] = trend["duration_bin"].apply(lambda x: x.mid)

    plt.figure(figsize=(11, 7))
    sns.lineplot(data=trend, x="mid", y="popularity", marker="o", linewidth=2.2)
    plt.title("Impact 1/5 — Duration Sweet Spot Curve\n(Median popularity by track length)")
    plt.xlabel("Track Length (minutes)")
    plt.ylabel("Median Popularity")
    save_fig("impact_1_duration_sweet_spot.png")

    best = trend.loc[trend["popularity"].idxmax()]
    findings.append(
        f"Impact 1/5: The duration sweet spot appears near {best['mid']:.2f} minutes with peak median popularity {best['popularity']:.1f}."
    )


def visual_extra_genre_popularity_spread(df: pd.DataFrame, findings: list[str]) -> None:
    req = {"track_genre", "popularity"}
    if not req.issubset(df.columns):
        return

    data = df.dropna(subset=list(req)).copy()
    if data.empty:
        return

    top_genres = data["track_genre"].value_counts().head(15).index.tolist()
    spread = data[data["track_genre"].isin(top_genres)].copy()
    if spread.empty:
        return

    plt.figure(figsize=(14, 8))
    sns.violinplot(
        data=spread,
        x="track_genre",
        y="popularity",
        order=top_genres,
        cut=0,
        inner=None,
        color="#7aa2ff",
    )
    sns.boxplot(
        data=spread,
        x="track_genre",
        y="popularity",
        order=top_genres,
        width=0.2,
        showfliers=False,
        boxprops={"facecolor": "white", "alpha": 0.75},
    )
    plt.title("Extra — Genre Popularity Spread\n(Violin + Box for Top 15 Genres by Track Count)")
    plt.xlabel("Genre")
    plt.ylabel("Popularity")
    plt.xticks(rotation=35, ha="right")
    save_fig("extra_genre_popularity_spread_top15.png")

    genre_stats = (
        spread.groupby("track_genre", as_index=False)
        .agg(
            median_pop=("popularity", "median"),
            q1=("popularity", lambda s: s.quantile(0.25)),
            q3=("popularity", lambda s: s.quantile(0.75)),
        )
        .assign(iqr=lambda d: d["q3"] - d["q1"])
    )
    highest_median = genre_stats.sort_values("median_pop", ascending=False).iloc[0]
    widest_spread = genre_stats.sort_values("iqr", ascending=False).iloc[0]
    findings.append(
        "Additional: Genre popularity spread (top 15 genres) shows "
        f"'{highest_median['track_genre']}' with the highest median popularity ({highest_median['median_pop']:.1f}), "
        f"while '{widest_spread['track_genre']}' has the widest interquartile spread ({widest_spread['iqr']:.1f})."
    )


def visual_extra_explicit_comparison(df: pd.DataFrame, findings: list[str]) -> None:
    req = {"explicit", "popularity", "track_genre"}
    if not req.issubset(df.columns):
        return

    data = df.dropna(subset=["explicit", "popularity"]).copy()
    if data.empty:
        return

    data["explicit_label"] = np.where(data["explicit"], "Explicit", "Non-explicit")

    overall = (
        data.groupby("explicit_label", as_index=False)
        .agg(median_pop=("popularity", "median"), tracks=("popularity", "size"))
        .sort_values("median_pop", ascending=False)
    )
    order = ["Non-explicit", "Explicit"]
    if not set(order).issubset(set(overall["explicit_label"])):
        order = overall["explicit_label"].tolist()

    top_genres = data["track_genre"].value_counts().head(10).index.tolist()
    by_genre = (
        data[data["track_genre"].isin(top_genres)]
        .groupby(["track_genre", "explicit_label"], as_index=False)
        .agg(median_pop=("popularity", "median"), tracks=("popularity", "size"))
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [1, 1.4]})

    sns.barplot(
        data=overall,
        x="explicit_label",
        y="median_pop",
        hue="explicit_label",
        order=order,
        ax=axes[0],
        palette="Set2",
        legend=False,
    )
    axes[0].set_title("Extra — Explicit vs Non-explicit Popularity\n(Overall Median Popularity)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Median Popularity")
    for _, row in overall.iterrows():
        axes[0].text(
            order.index(row["explicit_label"]) if row["explicit_label"] in order else 0,
            row["median_pop"] + 0.6,
            f"n={row['tracks']:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    if not by_genre.empty:
        sns.pointplot(
            data=by_genre,
            x="track_genre",
            y="median_pop",
            hue="explicit_label",
            order=top_genres,
            dodge=0.35,
            markers=["o", "s"],
            linestyles="-",
            ax=axes[1],
        )
        axes[1].set_title("Top 10 Genres: Median Popularity by Explicit Label")
        axes[1].set_xlabel("Genre")
        axes[1].set_ylabel("Median Popularity")
        axes[1].tick_params(axis="x", rotation=35)
        axes[1].legend(title="Track Type")
    else:
        axes[1].axis("off")

    save_fig("extra_explicit_vs_nonexplicit_popularity.png")

    explicit_med = overall.loc[overall["explicit_label"] == "Explicit", "median_pop"]
    non_exp_med = overall.loc[overall["explicit_label"] == "Non-explicit", "median_pop"]
    if not explicit_med.empty and not non_exp_med.empty:
        diff = explicit_med.iloc[0] - non_exp_med.iloc[0]
        lead = "Explicit" if diff > 0 else "Non-explicit"
        findings.append(
            "Additional: Explicit vs non-explicit comparison indicates "
            f"{lead} tracks have higher median popularity by {abs(diff):.1f} points overall."
        )


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    df = load_data(DATA_PATH)
    findings: list[str] = []

    visual_5_genre_fingerprint(df, findings)
    visual_4_hit_zone_heatmap(df, findings)
    visual_3_artist_consistency(df, findings)
    visual_2_tempo_loudness(df, findings)
    visual_1_duration_popularity(df, findings)
    visual_extra_genre_popularity_spread(df, findings)
    visual_extra_explicit_comparison(df, findings)

    findings_path = OUT_DIR / "findings_ranked_1_to_5.txt"
    findings_path.write_text("\n".join(findings) + "\n", encoding="utf-8")

    print("Generated files:")
    for file in sorted(OUT_DIR.glob("*.png")):
        print(f"- {file}")
    print(f"- {findings_path}")


if __name__ == "__main__":
    main()