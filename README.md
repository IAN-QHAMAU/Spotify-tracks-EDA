# Spotify Tracks EDA Story

A focused exploratory data analysis (EDA) project on a Spotify tracks dataset to discover what musical patterns are associated with track popularity.

## Live Project Page

Visit the published page here: [https://ian-qhamau.github.io/Spotify-tracks-EDA/](https://ian-qhamau.github.io/Spotify-tracks-EDA/)

## Project Snapshot

- **Dataset:** `data/spotify tracks dataset.csv`
- **Scale:** ~114k rows of tracks across many genres
- **Core question:** Which combinations of genre, mood, artist consistency, and audio features align with higher popularity?
- **Main output:** 7 visuals (5 ranked + 2 comparative) + findings text in `visualisations/outputs/`

## Repository Structure

```text
music/
├── data/
│   └── spotify tracks dataset.csv
├── notebooks/
│   └── music.ipynb
├── visualisations/
│   ├── spotify_unique_visuals.py
│   ├── visuals.ipynb
│   └── outputs/
│       ├── extra_explicit_vs_nonexplicit_popularity.png
│       ├── extra_genre_popularity_spread_top15.png
│       ├── impact_1_duration_sweet_spot.png
│       ├── impact_2_tempo_loudness_hit_terrain.png
│       ├── impact_3_artist_consistency.png
│       ├── impact_4_hit_zone_heatmap.png
│       ├── impact_5_genre_fingerprint.png
│       └── findings_ranked_1_to_5.txt
└── README.md
```

## Data Fields Used

The analysis primarily uses:

- `popularity`
- `duration_ms` (converted to `duration_min`)
- `danceability`, `energy`, `valence`
- `tempo`, `loudness`
- `artists`, `track_genre`
- Supporting fields for potential extensions: `explicit`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`

## Methodology

1. Load and clean data (`Unnamed` columns removed, numeric coercion on audio/popularity fields).
2. Engineer `duration_min` from `duration_ms`.
3. Build five impact-ranked visualization views.
4. Build two comparative visuals: genre spread and explicit-vs-non-explicit performance.
5. Write concise findings to `visualisations/outputs/findings_ranked_1_to_5.txt`.

## Ranked Findings (Most to Least Impact)

1. **Genre fingerprint map (Impact 5/5):** popularity clusters can be interpreted by genre mood position (energy × valence).
2. **Hit zone heatmap (Impact 4/5):** best popularity regions appear in specific danceability/energy bins.
3. **Artist consistency map (Impact 3/5):** some artists combine high average popularity with low volatility.
4. **Tempo–loudness terrain (Impact 2/5):** higher-performing tracks concentrate near a central BPM/loudness region.
5. **Duration sweet spot (Impact 1/5):** popularity peaks around a mid-length track duration band.

## Visual Outputs

Open these generated images in `visualisations/outputs/`:

- `extra_genre_popularity_spread_top15.png`
- `extra_explicit_vs_nonexplicit_popularity.png`
- `impact_5_genre_fingerprint.png`
- `impact_4_hit_zone_heatmap.png`
- `impact_3_artist_consistency.png`
- `impact_2_tempo_loudness_hit_terrain.png`
- `impact_1_duration_sweet_spot.png`

## Additional Comparative Findings

- **Genre popularity spread (top 15 genres):** `pop` has the highest median popularity (**66.0**) and also the widest interquartile spread (**69.0**), indicating both high upside and high dispersion.
- **Explicit vs non-explicit comparison:** explicit tracks have higher median popularity by **3.0 points** overall.

Companion insights:

- `findings_ranked_1_to_5.txt`

## How to Regenerate the Analysis

From project root:

```bash
python visualisations/spotify_unique_visuals.py
```

The script will regenerate all PNGs and refresh `findings_ranked_1_to_5.txt`.

## Why This EDA Is Useful

This EDA is designed for decision support, not just plotting:

- **A&R / talent strategy:** identify artists with reliable performance profiles
- **Production direction:** infer audio feature zones linked to better outcomes
- **Genre strategy:** compare market behavior by genre mood fingerprint
- **Release optimization:** reason about track duration and sonic positioning

## Notes

- Popularity is platform-derived and should be interpreted as a relative signal.
- Results are correlational, not causal.
- Segmenting by era/region can materially change conclusions.
