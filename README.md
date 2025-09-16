## POI Extraction from Google Maps

This tool collects Points of Interest (POIs) from Google Maps for GeoJSON extents, retrieves full metadata for each POI via the official Google Places API (New, v1), and exports an Excel workbook with one sheet per input file.

### Features
- Grid-tiles the area (per municipal polygon) and runs v1 Nearby Search (`places:searchNearby`) to cover POIs
- Fetches v1 Place Details (`places/{place_id}`) requesting full resource via `X-Goog-FieldMask: *`
- Deduplicates across grid overlaps
- Exports one Excel sheet per municipality with flattened metadata
 - Optional polygon coverage: supply municipal polygons to ensure the grid covers 100% of each municipality and ignores outside areas

### Requirements
- Python 3.9+
- A Google Maps Platform API key with Places API (New) enabled

### Setup
1. Create and restrict an API key in Google Cloud Console with Places API enabled.
2. Clone this repository.
3. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4. Configure your environment variables (copy the template):

```bash
cp env.example .env
```

Edit `.env` to set `GOOGLE_MAPS_API_KEY`.

### Input
- Provide one GeoJSON per extent (recommended), or a directory of GeoJSON files. Each file becomes one output sheet named after the file (without extension). Files may contain multiple polygons; they will be unioned. You can also pass a single FeatureCollection file (this will produce a single sheet named after the file).

### Usage

```bash
# Single FeatureCollection (supported, produces one sheet)
python poi_extraction.py \
  --output pois.xlsx \
  --radius-m 1500 \
  --extent-geojson data/municipalities.geojson

# Multiple files (one per extent, multi-polygons allowed)
python poi_extraction.py \
  --output pois.xlsx \
  --radius-m 1500 \
  --extent-geojson data/misiliscemi.geojson data/valderice.geojson data/vita.geojson data/buseto.geojson

# Directory of files
python poi_extraction.py \
  --output pois.xlsx \
  --radius-m 1500 \
  --extent-geojson data/muni_geojsons/
```

Minimal FeatureCollection structure (example):

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": { "name": "Misiliscemi" },
      "geometry": { "type": "Polygon", "coordinates": [[[12.5,37.9],[12.6,37.9],[12.6,38.0],[12.5,38.0],[12.5,37.9]]] }
    }
  ]
}
```

Key arguments:
- `--output`: output Excel path (default: `pois.xlsx`)
- `--radius-m`: Nearby Search radius in meters per grid point (default: 1500)
- `--grid-overlap`: Overlap factor for grid spacing (default: 1.3). Effective spacing is `radius_m / grid_overlap`.
- `--max-workers`: Max parallel requests for Place Details (default: 10)
- `--extent-geojson`: One or more GeoJSON paths or a directory; each file becomes one sheet (supports multi-polygons)

### Notes and Limits
- This tool uses the official Google Places API (New, v1). Ensure you comply with Google Maps Platform Terms of Service and your billing/quota limits.
- Nearby Search caps results per request. A grid is used to improve coverage; exact completeness depends on your chosen radius/overlap and Google’s indexing.
- The script requests an extensive set of Place Details fields and serializes nested structures (lists/dicts) as JSON strings so that "all metadata" is preserved in the spreadsheet.

### Output
- An Excel workbook where each input file becomes its own sheet. Columns are derived from flattened Place Details (nested objects/arrays are JSON strings).

#### Sheet naming and file mapping
- When you pass multiple files or a directory via `--extent-geojson`, the tool creates one sheet per file.
- The sheet name equals the file name without extension (e.g., `misiliscemi.geojson` → sheet `misiliscemi`).
- Each file can contain multiple polygons; they are unioned before gridding to ensure full coverage for that sheet.

#### RunInfo sheet
- The workbook includes a `RunInfo` sheet with metadata: start/end time, duration, number of files, radius, overlap, type filter, language, worker count, API base, Python/pandas/requests versions, Shapely availability, sources, and a per-sheet summary table (grid points, place_ids collected, details fetched).

### Troubleshooting
- If you see `REQUEST_DENIED` or `OVER_QUERY_LIMIT`, confirm your key, quotas, and API enablement. The script retries with exponential backoff for transient errors.

### Converting shapefiles to GeoJSON

If your polygons are in a shapefile, export to GeoJSON first. Two options:

1) Using GDAL/ogr2ogr:

```bash
ogr2ogr -f GeoJSON data/municipalities.geojson path/to/your.shp
```

2) Using GeoPandas (Python):

```python
import geopandas as gpd
gdf = gpd.read_file('path/to/your.shp')
gdf.to_file('data/municipalities.geojson', driver='GeoJSON')
```


