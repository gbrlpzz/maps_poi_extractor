import argparse
import concurrent.futures
import json
import math
import os
import sys
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm
try:
    from shapely.geometry import Point, shape, mapping
    from shapely.ops import unary_union
    from shapely.prepared import prep
except Exception:
    Point = None
    shape = None
    mapping = None
    unary_union = None
    prep = None


PLACES_V1_BASE = "https://places.googleapis.com/v1"


@dataclass
class Bounds:
    north: float
    south: float
    east: float
    west: float


def meters_per_degree_latitude() -> float:
    return 111_320.0


def meters_per_degree_longitude(latitude_deg: float) -> float:
    return 111_320.0 * math.cos(math.radians(latitude_deg))


def load_api_key() -> str:
    load_dotenv(override=False)
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_MAPS_API_KEY is not set. Create a .env from env.example.", file=sys.stderr)
        sys.exit(1)
    return api_key


def safe_sheet_name(name: str) -> str:
    # Excel sheet names max 31 chars, cannot contain: : \ / ? * [ ]
    invalid = ":\\/?*[]"
    for ch in invalid:
        name = name.replace(ch, "_")
    return name[:31]


class HttpError(Exception):
    pass


def generate_grid_points(bounds: Bounds, radius_m: int, overlap_factor: float) -> List[Tuple[float, float]]:
    # Effective spacing smaller than radius to ensure overlap
    spacing_m = max(50.0, radius_m / max(1.0, overlap_factor))
    points: List[Tuple[float, float]] = []
    # Traverse rows by latitude
    lat = bounds.south
    while lat <= bounds.north:
        m_per_deg_lon = max(1e-6, meters_per_degree_longitude(lat))
        dlat = spacing_m / meters_per_degree_latitude()
        dlon = spacing_m / m_per_deg_lon
        lon = bounds.west
        while lon <= bounds.east:
            points.append((lat, lon))
            lon += dlon
        lat += dlat
    return points


def filter_points_in_polygon(points: List[Tuple[float, float]], polygon: object) -> List[Tuple[float, float]]:
    if not Point:
        return points
    if hasattr(polygon, "contains"):
        geom = polygon  # shapely geometry
    else:
        if not shape:
            return points
        geom = shape(polygon)  # GeoJSON
    prepared = prep(geom)
    filtered: List[Tuple[float, float]] = []
    for lat, lon in points:
        if prepared.contains(Point(lon, lat)):
            filtered.append((lat, lon))
    return filtered


def bounds_from_geojson_geometry(geom: Dict) -> Bounds:
    if shape:
        s = shape(geom)
        minx, miny, maxx, maxy = s.bounds
        return Bounds(north=maxy, south=miny, east=maxx, west=minx)
    # Fallback: compute by traversing coordinates
    def walk_coords(g: Dict) -> List[Tuple[float, float]]:
        coords: List[Tuple[float, float]] = []
        t = g.get("type")
        c = g.get("coordinates", [])
        if t == "Polygon":
            for ring in c:
                for x, y in ring:
                    coords.append((x, y))
        elif t == "MultiPolygon":
            for poly in c:
                for ring in poly:
                    for x, y in ring:
                        coords.append((x, y))
        else:
            raise ValueError(f"Unsupported geometry type: {t}")
        return coords

    coords = walk_coords(geom)
    if not coords:
        raise ValueError("Empty geometry coordinates")
    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]
    return Bounds(north=max(ys), south=min(ys), east=max(xs), west=min(xs))


def load_municipalities_from_paths(paths: List[str]) -> List[Tuple[str, object]]:
    municipalities: List[Tuple[str, object]] = []
    for path in paths:
        if os.path.isdir(path):
            for fname in sorted(os.listdir(path)):
                if not fname.lower().endswith((".geojson", ".json")):
                    continue
                municipalities.extend(load_municipalities_from_paths([os.path.join(path, fname)]))
            continue

        with open(path, "r", encoding="utf-8") as f:
            gj = json.load(f)

        filename_name = os.path.splitext(os.path.basename(path))[0]

        # Accept FeatureCollection, Feature, or bare geometry
        features = []
        if isinstance(gj, dict) and gj.get("type") == "FeatureCollection":
            features = gj.get("features", [])
        elif isinstance(gj, dict) and gj.get("type") == "Feature":
            features = [gj]
        else:
            # assume geometry object
            features = [{"type": "Feature", "properties": {}, "geometry": gj}]

        geometries = []
        for feat in features:
            geom = feat.get("geometry")
            if not geom:
                continue
            if shape:
                try:
                    geometries.append(shape(geom))
                except Exception:
                    pass
            else:
                geometries.append(geom)

        muni_name = filename_name

        if shape and geometries and hasattr(geometries[0], "geom_type"):
            # Shapely geometries available; union them into a single geometry
            try:
                unioned = unary_union(geometries) if unary_union else geometries[0]
            except Exception:
                unioned = geometries[0]
            municipalities.append((muni_name, unioned))
        else:
            # Fall back to MultiPolygon GeoJSON
            if not geometries:
                continue
            if len(geometries) == 1 and isinstance(geometries[0], dict):
                out_geom = geometries[0]
            else:
                # Compose MultiPolygon from polygons
                coords: List[List[List[Tuple[float, float]]]] = []
                for g in geometries:
                    if isinstance(g, dict) and g.get("type") == "Polygon":
                        coords.append(g.get("coordinates", []))
                out_geom = {"type": "MultiPolygon", "coordinates": [c for c in coords if c]}
            municipalities.append((muni_name, out_geom))

    return municipalities

def iter_nearby_place_ids(session: requests.Session, api_key: str, lat: float, lon: float, radius_m: int, type_filter: Optional[str] = None) -> Iterable[str]:
    url = f"{PLACES_V1_BASE}/places:searchNearby"
    headers = {
        "X-Goog-Api-Key": api_key,
        # Use broad field mask for robustness; can be narrowed later
        "X-Goog-FieldMask": "*",
        "Content-Type": "application/json",
    }
    body: Dict = {
        "locationRestriction": {
            "circle": {
                "center": {"latitude": lat, "longitude": lon},
                "radius": float(radius_m),
            }
        },
        # For v1 Nearby Search, use maxResultCount instead of pageSize
        "maxResultCount": 20,
    }
    if type_filter:
        body["includedTypes"] = [type_filter]

    next_page_token: Optional[str] = None
    while True:
        payload = dict(body)
        if next_page_token:
            payload["pageToken"] = next_page_token
        resp = session.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code in (429, 500, 503):
            # transient
            time.sleep(2.0)
            continue
        if not resp.ok:
            try:
                err = resp.json()
            except Exception:
                err = {"status_code": resp.status_code, "text": resp.text[:300]}
            print(f"ERROR: Nearby Search failed: {err}", file=sys.stderr)
            return
        data = resp.json()
        if isinstance(data, dict) and data.get("error"):
            print(f"ERROR: Nearby Search error: {data['error']}", file=sys.stderr)
            return
        for place in data.get("places", []):
            # Prefer resource name 'places/{place_id}'
            resource_name = place.get("name")
            if resource_name and "/" in resource_name:
                yield resource_name.rsplit("/", 1)[-1]
                continue
            pid = place.get("id") or place.get("placeId")
            if pid:
                yield pid
        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            return


DETAIL_FIELDS: List[str] = [
    # Basic
    "address_components",
    "adr_address",
    "business_status",
    "formatted_address",
    "geometry",
    "icon",
    "icon_background_color",
    "icon_mask_base_uri",
    "name",
    "place_id",
    "plus_code",
    "types",
    "url",
    "utc_offset",
    # Contact
    "current_opening_hours",
    "opening_hours",
    "secondary_opening_hours",
    "website",
    "formatted_phone_number",
    "international_phone_number",
    # Atmosphere
    "price_level",
    "rating",
    "user_ratings_total",
    "reviews",
    "editorial_summary",
    # Attributes
    "curbside_pickup",
    "delivery",
    "dine_in",
    "reservable",
    "serves_breakfast",
    "serves_brunch",
    "serves_lunch",
    "serves_dinner",
    "serves_beer",
    "serves_wine",
    "serves_cocktails",
    "serves_coffee",
    "takeout",
    "wheelchair_accessible_entrance",
    # Media
    "photos",
]


@retry(wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(5),
       retry=(retry_if_exception_type(HttpError)))
def fetch_place_details(session: requests.Session, api_key: str, place_id: str, language: Optional[str] = None) -> Optional[Dict]:
    url = f"{PLACES_V1_BASE}/places/{place_id}"
    headers = {
        "X-Goog-Api-Key": api_key,
        # Request the full Place resource; downstream we flatten to preserve all metadata
        "X-Goog-FieldMask": "*",
    }
    params = {}
    if language:
        params["languageCode"] = language
    resp = session.get(url, headers=headers, params=params, timeout=30)
    if resp.status_code in (429, 500, 503):
        raise HttpError(f"Transient error {resp.status_code} from Places Details")
    if resp.status_code == 404:
        return None
    if not resp.ok:
        # Non-retryable error
        return None
    return resp.json()


def flatten_value(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    # For lists/dicts, preserve full content as JSON string
    return json.dumps(value, ensure_ascii=False)


def flatten_place_detail(detail: Dict) -> Dict[str, object]:
    flat: Dict[str, object] = {}

    def _walk(prefix: str, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    _walk(key, v)
                elif isinstance(v, list):
                    # store list as JSON to preserve all content
                    flat[key] = json.dumps(v, ensure_ascii=False)
                else:
                    flat[key] = v
        elif isinstance(obj, list):
            flat[prefix] = json.dumps(obj, ensure_ascii=False)
        else:
            flat[prefix] = obj

    _walk("", detail)
    # Ensure some top-level fields always exist
    # Try to ensure a few identifiers are present across API versions
    if "place_id" not in flat:
        # From v1 resource name: places/{place_id}
        resource_name = detail.get("name")
        if isinstance(resource_name, str) and "/" in resource_name:
            flat["place_id"] = resource_name.rsplit("/", 1)[-1]
    for must_have in ("name", "place_id", "formatted_address", "formattedAddress"):
        flat.setdefault(must_have, detail.get(must_have))
    return flat


def collect_place_ids_for_bounds(session: requests.Session, api_key: str, bounds: Bounds, radius_m: int, overlap_factor: float, type_filter: Optional[str], progress_desc: str, polygon_geojson: Optional[object] = None) -> Tuple[Set[str], int]:
    points = generate_grid_points(bounds, radius_m=radius_m, overlap_factor=overlap_factor)
    if polygon_geojson:
        points = filter_points_in_polygon(points, polygon_geojson)
    place_ids: Set[str] = set()
    for lat, lon in tqdm(points, desc=progress_desc, unit="pt", leave=False):
        for pid in iter_nearby_place_ids(session, api_key, lat, lon, radius_m, type_filter=type_filter):
            place_ids.add(pid)
    return place_ids, len(points)


def read_municipalities_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    return names


def main():
    parser = argparse.ArgumentParser(description="Collect POIs from Google Maps for municipalities defined by polygons and export to Excel")
    parser.add_argument("--output", default="pois.xlsx", help="Output Excel workbook path")
    parser.add_argument("--radius-m", type=int, default=1500, help="Nearby Search radius in meters per grid point")
    parser.add_argument("--grid-overlap", type=float, default=1.3, help="Overlap factor; effective spacing is radius_m / overlap")
    parser.add_argument("--country", default=os.getenv("DEFAULT_COUNTRY"), help="Unused with polygons; kept for compatibility")
    parser.add_argument("--type", default=None, help="Optional Places type filter (e.g., restaurant); omit for all types")
    parser.add_argument("--max-workers", type=int, default=10, help="Parallel workers for Place Details requests")
    parser.add_argument("--extent-geojson", required=True, nargs='+', help="One or more GeoJSON paths or a directory; each file becomes one sheet (supports multi-polygons)")
    parser.add_argument("--language", default=None, help="Optional language code for details (e.g., it)")
    args = parser.parse_args()

    api_key = load_api_key()

    session = requests.Session()

    # Load extent polygons
    muni_geoms = load_municipalities_from_paths(args.extent_geojson)
    if not muni_geoms:
        print("No municipal geometries found in provided path(s).", file=sys.stderr)
        sys.exit(1)

    # Prepare Excel writer
    writer = pd.ExcelWriter(args.output, engine="openpyxl")
    # Ensure workbook has at least one sheet (metadata) to prevent empty workbook error on early failures
    started_at = datetime.now()
    sources_list = [os.path.basename(p) for p in (args.extent_geojson if isinstance(args.extent_geojson, list) else [args.extent_geojson])]
    pd.DataFrame([
        {"note": "Run initializing", "started_at": started_at.isoformat(), "sources": ", ".join(sources_list)}
    ]).to_excel(writer, sheet_name="RunInfo", index=False)

    summary_rows: List[Dict[str, object]] = []

    try:
        for name, geom in tqdm(muni_geoms, desc="Sheets", unit="sheet"):
            # Bounds from geometry
            try:
                if isinstance(geom, dict):
                    bounds = bounds_from_geojson_geometry(geom)
                else:
                    minx, miny, maxx, maxy = geom.bounds
                    bounds = Bounds(north=maxy, south=miny, east=maxx, west=minx)
            except Exception as e:
                pd.DataFrame([{"error": f"Invalid geometry: {e}"}]).to_excel(writer, sheet_name=safe_sheet_name(name), index=False)
                continue

            # Collect place_ids via grid of Nearby Search filtered to polygon
            place_ids, grid_points = collect_place_ids_for_bounds(
                session,
                api_key,
                bounds,
                radius_m=args.radius_m,
                overlap_factor=args.grid_overlap,
                type_filter=args.type,
                progress_desc=f"Grid {name}",
                polygon_geojson=geom,
            )

            if not place_ids:
                print(f"INFO: No places found for '{name}'.", file=sys.stderr)
                # Write an empty sheet to keep structure
                pd.DataFrame().to_excel(writer, sheet_name=safe_sheet_name(name), index=False)
                summary_rows.append({
                    "sheet": safe_sheet_name(name),
                    "source": name,
                    "grid_points": grid_points,
                    "place_ids": 0,
                    "details": 0,
                })
                continue

            # Fetch details in parallel
            details_list: List[Dict] = []

            def _fetch(pid: str) -> Optional[Dict]:
                try:
                    return fetch_place_details(session, api_key, pid, language=args.language)
                except Exception as e:
                    return None

            with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
                for result in tqdm(pool.map(_fetch, place_ids), total=len(place_ids), desc=f"Details {name}", unit="pl", leave=False):
                    if result:
                        details_list.append(result)

            if not details_list:
                pd.DataFrame().to_excel(writer, sheet_name=safe_sheet_name(name), index=False)
                summary_rows.append({
                    "sheet": safe_sheet_name(name),
                    "source": name,
                    "grid_points": grid_points,
                    "place_ids": len(place_ids),
                    "details": 0,
                })
                continue

            # Flatten and export
            rows = [flatten_place_detail(d) for d in details_list]
            df = pd.DataFrame(rows)
            # Sort columns alphabetically for consistency
            df = df.reindex(sorted(df.columns), axis=1)
            df.to_excel(writer, sheet_name=safe_sheet_name(name), index=False)

            summary_rows.append({
                "sheet": safe_sheet_name(name),
                "source": name,
                "grid_points": grid_points,
                "place_ids": len(place_ids),
                "details": len(details_list),
            })

    finally:
        # Replace RunInfo with full metadata
        finished_at = datetime.now()
        duration_s = (finished_at - started_at).total_seconds()
        meta = {
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "duration_seconds": int(duration_s),
            "files_count": len(muni_geoms),
            "radius_m": args.radius_m,
            "grid_overlap": args.grid_overlap,
            "type_filter": args.type or "",
            "language": args.language or "",
            "max_workers": args.max_workers,
            "api_base": PLACES_V1_BASE,
            "python_version": sys.version.split(" ")[0],
            "pandas_version": pd.__version__,
            "requests_version": requests.__version__,
            "shapely_available": bool(shape),
            "sources": ", ".join(sources_list),
        }
        # Remove existing RunInfo sheet if present
        try:
            book = writer.book
            if "RunInfo" in book.sheetnames:
                std = book["RunInfo"]
                book.remove(std)
        except Exception:
            pass
        meta_df = pd.DataFrame([meta])
        meta_df.to_excel(writer, sheet_name="RunInfo", index=False)
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_excel(writer, sheet_name="RunInfo", index=False, startrow=meta_df.shape[0] + 2)
        writer.close()

    print(f"Done. Wrote {args.output}")


if __name__ == "__main__":
    main()


