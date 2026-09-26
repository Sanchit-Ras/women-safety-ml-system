"""The overlap table is evidence, so its arithmetic is pinned here on unit squares. No real file is read."""
import csv
import json

from shapely.geometry import shape

from data_ingestion.geometry import (AREA_NOTE, LGD_LIST_COLUMNS, OVERLAP_COLUMNS, checks, load_new, load_new_from,
                                     load_old, load_old_from, overlap, read_overlap, simplify, simplify_within,
                                     write_lgd_list, write_overlap)


def square(x0, y0, x1, y1):
    return {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}


def old_fc(*items):
    """(state, name, geometry) triples shaped like the GADM file."""
    return {"type": "FeatureCollection",
            "features": [{"type": "Feature", "properties": {"NAME_1": s, "NAME_2": n}, "geometry": g} for s, n, g in items]}


def new_fc(*items):
    """(code, name, state_code, state_name, geometry) tuples shaped like the LGD file."""
    return {"type": "FeatureCollection",
            "features": [{"type": "Feature",
                          "properties": {"dist_lgd": c, "dtname": n, "state_lgd": sc, "stname": sn, "OBJECTID": 1},
                          "geometry": g} for c, n, sc, sn, g in items]}


def by_code(df):
    return {int(r.new_code): r for r in df.itertuples()}


# ---- loading -------------------------------------------------------------------------------

def test_load_old_applies_the_old_map_state_aliases_after_norm_state():
    old = load_old_from(old_fc(("Orissa", "Koraput", square(0, 0, 1, 1)),
                               ("Uttaranchal", "Nainital", square(2, 0, 3, 1)),
                               ("Andaman and Nicobar", "Andaman Islands", square(4, 0, 5, 1)),
                               ("Andhra Pradesh", "Y.S.R.", square(6, 0, 7, 1))))
    assert [o["state_norm"] for o in old] == ["odisha", "uttarakhand", "andaman and nicobar islands", "andhra pradesh"]
    assert [o["name_norm"] for o in old] == ["koraput", "nainital", "andaman islands", "ysr"]
    assert old[0]["state"] == "Orissa" and old[0]["name"] == "Koraput"          # source spellings kept for display


def test_load_old_makes_a_self_crossing_polygon_valid():
    bow_tie = {"type": "Polygon", "coordinates": [[[0, 0], [2, 2], [2, 0], [0, 2], [0, 0]]]}
    assert not shape(bow_tie).is_valid
    geom = load_old_from(old_fc(("Bihar", "Patna", bow_tie)))[0]["geom"]
    assert geom.is_valid and geom.area == 2.0                                   # two triangles of area 1


def test_load_keeps_only_the_polygon_when_make_valid_cuts_a_spike_off():
    spiked = {"type": "Polygon", "coordinates": [[[0, 0], [2, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}   # spike (1,0)->(2,0)
    assert not shape(spiked).is_valid
    geom = load_new_from(new_fc((1, "Patna", 10, "BIHAR", spiked)))[0]["geom"]
    assert geom.geom_type in ("Polygon", "MultiPolygon") and geom.is_valid and geom.area == 1.0


def collinear_ring():
    """A 'polygon' whose vertices all lie on one line: make_valid turns it into a line, not a polygon."""
    return {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [2, 0], [0, 0]]]}


def test_load_reduces_a_collinear_ring_to_an_empty_polygon_never_a_line():
    for geom in (load_old_from(old_fc(("Bihar", "Patna", collinear_ring())))[0]["geom"],
                 load_new_from(new_fc((1, "Patna", 10, "BIHAR", collinear_ring())))[0]["geom"]):
        assert geom.geom_type in ("Polygon", "MultiPolygon") and geom.is_empty and geom.area == 0


def test_a_ring_with_no_area_gets_no_rows_and_is_named_by_checks():
    old = load_old_from(old_fc(("Bihar", "Patna", collinear_ring())))
    new = load_new_from(new_fc((1, "Patna", 10, "BIHAR", square(0, 0, 1, 1))))
    df = overlap(old, new, min_share=0)
    assert len(df) == 0
    assert checks(df, old, new)["old_without_overlap"] == [{"state": "Bihar", "name": "Patna"}]


def test_load_new_reads_codes_as_ints_and_keeps_only_what_the_pipeline_uses():
    new = load_new_from(new_fc((502, "Anantapur", 28, "ANDHRA PRADESH", square(0, 0, 1, 1))))
    fields = {k: v for k, v in new[0].items() if k != "geom"}
    assert fields == {"code": 502, "name": "Anantapur", "state_code": 28, "state_name": "ANDHRA PRADESH",
                      "name_norm": "anantapur"}
    assert new[0]["geom"].area == 1.0


def test_load_old_and_load_new_read_a_geojson_file(tmp_path):
    old_path, new_path = tmp_path / "old.geojson", tmp_path / "new.geojson"
    old_path.write_text(json.dumps(old_fc(("Orissa", "Koraput", square(0, 0, 1, 1)))), encoding="utf-8")
    new_path.write_text(json.dumps(new_fc((375, "Koraput", 21, "ODISHA", square(0, 0, 1, 1)))), encoding="utf-8")
    assert load_old(old_path)[0]["state_norm"] == "odisha"
    assert load_new(new_path)[0]["code"] == 375


# ---- overlap --------------------------------------------------------------------------------

def test_an_old_square_split_evenly_across_two_new_squares():
    old = load_old_from(old_fc(("Andhra Pradesh", "Warangal", square(0, 0, 2, 1))))      # area 2
    new = load_new_from(new_fc((519, "Warangal Urban", 36, "TELANGANA", square(0, 0, 1, 1)),    # area 1, fully inside
                               (520, "Warangal Rural", 36, "TELANGANA", square(1, 0, 3, 1))))   # area 2, half inside
    df = overlap(old, new)
    assert list(df.columns) == OVERLAP_COLUMNS and len(df) == 2
    rows = by_code(df)
    assert rows[519].share == 0.5 and rows[519].new_share == 1.0
    assert rows[520].share == 0.5 and rows[520].new_share == 0.5
    assert (rows[519].old_state, rows[519].old_name, rows[519].old_norm, rows[519].old_state_norm,
            rows[519].new_name, rows[519].new_state_code) == ("Andhra Pradesh", "Warangal", "warangal", "andhra pradesh",
                                                              "Warangal Urban", 36)


def test_a_renamed_district_is_one_row_with_share_one():
    old = load_old_from(old_fc(("Andhra Pradesh", "Cuddapah", square(5, 5, 6, 6))))
    new = load_new_from(new_fc((503, "Y.S.R.", 28, "ANDHRA PRADESH", square(5, 5, 6, 6)),
                               (504, "Kurnool", 28, "ANDHRA PRADESH", square(6, 5, 7, 6))))    # touches only along an edge
    df = overlap(old, new)
    assert len(df) == 1
    assert (df.loc[0, "new_code"], df.loc[0, "share"], df.loc[0, "new_share"]) == (503, 1.0, 1.0)


def test_an_edge_touching_neighbour_is_never_a_row_even_with_min_share_zero():
    old = load_old_from(old_fc(("Andhra Pradesh", "Cuddapah", square(5, 5, 6, 6))))
    new = load_new_from(new_fc((503, "Y.S.R.", 28, "ANDHRA PRADESH", square(5, 5, 6, 6)),
                               (504, "Kurnool", 28, "ANDHRA PRADESH", square(6, 5, 7, 6))))    # shares an edge, no land
    df = overlap(old, new, min_share=0)
    assert df["new_code"].tolist() == [503]                                     # no share-0.0 row for Kurnool
    assert checks(df, old, new)["new_without_cover"] == [{"code": 504, "name": "Kurnool", "state_name": "ANDHRA PRADESH"}]


def test_min_share_drops_boundary_slivers():
    old = load_old_from(old_fc(("Bihar", "Patna", square(0, 0, 1, 1))))
    new = load_new_from(new_fc((1, "Patna", 10, "BIHAR", square(0, 0, 0.995, 1)),
                               (2, "Nalanda", 10, "BIHAR", square(0.995, 0, 2, 1))))         # 0.5% of Patna
    assert by_code(overlap(old, new)).keys() == {1}
    kept = by_code(overlap(old, new, min_share=0))
    assert kept.keys() == {1, 2} and abs(kept[2].share - 0.005) < 1e-9


def test_rows_are_sorted_by_old_state_old_name_then_share_descending():
    old = load_old_from(old_fc(("Bihar", "Patna", square(0, 0, 1, 1)),
                               ("Assam", "Kamrup", square(10, 0, 11, 1))))
    new = load_new_from(new_fc((1, "A", 10, "BIHAR", square(0, 0, 0.3, 1)),
                               (2, "B", 10, "BIHAR", square(0.3, 0, 1, 1)),
                               (3, "C", 18, "ASSAM", square(10, 0, 11, 1))))
    df = overlap(old, new)
    assert df["old_state"].tolist() == ["Assam", "Bihar", "Bihar"]
    assert df["new_code"].tolist() == [3, 2, 1] and df["share"].tolist() == [1.0, 0.7, 0.3]


def test_overlap_with_no_old_or_no_new_polygons_is_an_empty_table_with_the_columns():
    new = load_new_from(new_fc((1, "A", 10, "BIHAR", square(0, 0, 1, 1))))
    old = load_old_from(old_fc(("Bihar", "Patna", square(0, 0, 1, 1))))
    assert list(overlap([], new).columns) == OVERLAP_COLUMNS and len(overlap([], new)) == 0
    assert list(overlap(old, []).columns) == OVERLAP_COLUMNS and len(overlap(old, [])) == 0


# ---- checks ------------------------------------------------------------------------------------

def test_checks_report_share_sums_and_the_polygons_that_never_appear():
    old = load_old_from(old_fc(("Bihar", "Patna", square(0, 0, 1, 1)),           # fully covered: sum 1.0
                               ("Bihar", "Gaya", square(2, 0, 4, 1)),            # half covered: sum 0.5
                               ("Rajasthan", "Jalor", square(6, 0, 7, 1)),       # two modern polygons overlap: sum 1.5
                               ("Assam", "Kamrup", square(50, 50, 51, 51))))     # nothing under it
    new = load_new_from(new_fc((1, "Patna", 10, "BIHAR", square(0, 0, 1, 1)),
                               (2, "Gaya", 10, "BIHAR", square(2, 0, 3, 1)),
                               (104, "Jalore", 8, "RAJASTHAN", square(6, 0, 7, 1)),          # not cut when Sanchor
                               (779, "Sanchor", 8, "RAJASTHAN", square(6.5, 0, 7, 1)),       # was carved out of it
                               (3, "Dibrugarh", 18, "ASSAM", square(80, 0, 81, 1))))   # nothing over it
    df = overlap(old, new)
    out = checks(df, old, new)
    assert (out["old_polygons"], out["old_with_rows"]) == (4, 3)
    assert out["share_sum"] == {"min": 0.5, "max": 1.5, "below_0.95": 1, "above_1.05": 1}
    assert out["old_without_overlap"] == [{"state": "Assam", "name": "Kamrup"}]
    assert out["new_without_cover"] == [{"code": 3, "name": "Dibrugarh", "state_name": "ASSAM"}]
    json.dumps(out)                                                              # it goes into the meta file as is


def test_checks_on_an_empty_table():
    out = checks(overlap([], []))
    assert out["share_sum"] == {"min": None, "max": None, "below_0.95": 0, "above_1.05": 0}
    assert out["old_without_overlap"] == [] and out["new_without_cover"] == []


def test_write_overlap_writes_the_table_and_a_meta_file_with_the_degree_note(tmp_path):
    old = load_old_from(old_fc(("Orissa", "Koraput", square(0, 0, 2, 1))))
    new = load_new_from(new_fc((375, "Koraput", 21, "ODISHA", square(0, 0, 1, 1)),
                               (376, "Malkangiri", 21, "ODISHA", square(1, 0, 2, 1))))
    csv_path, meta_path = tmp_path / "overlap.csv", tmp_path / "meta.json"
    df, meta = write_overlap(old, new, 0.01, out_path=csv_path, meta_path=meta_path)
    back = read_overlap(csv_path)
    assert list(back.columns) == OVERLAP_COLUMNS and back["share"].tolist() == [0.5, 0.5]
    assert back["new_code"].tolist() == df["new_code"].tolist() and back["old_state_norm"].tolist() == ["odisha"] * 2
    saved = json.loads(meta_path.read_text(encoding="utf-8"))
    assert saved["area_note"] == AREA_NOTE and saved["min_share"] == 0.01
    assert (saved["rows"], saved["old_polygons"], saved["new_polygons"]) == (2, 1, 2)
    assert saved["checks"]["share_sum"]["max"] == 1.0 and saved["checks"]["old_without_overlap"] == []
    assert len(saved["generated"]) == 10                                         # an ISO date


# ---- exports ------------------------------------------------------------------------------------

def test_lgd_list_flags_codes_that_appear_in_the_registry_and_sorts_by_state_then_code(tmp_path):
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"districts": [{"district_code": 502}, {"district_code": 1}]}), encoding="utf-8")
    new = load_new_from(new_fc((999, "Brand New", 28, "ANDHRA PRADESH", square(0, 0, 1, 1)),
                               (502, "Anantapur", 28, "ANDHRA PRADESH", square(1, 0, 2, 1)),
                               (1, "Anantnag", 1, "JAMMU & KASHMIR", square(2, 0, 3, 1))))
    out = tmp_path / "lgd.csv"
    assert write_lgd_list(new, registry, out) == 3
    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == LGD_LIST_COLUMNS
    assert [(r["state_code"], r["district_code"], r["name"], r["in_files"]) for r in rows] == [
        ("1", "1", "Anantnag", "1"), ("28", "502", "Anantapur", "1"), ("28", "999", "Brand New", "0")]
    assert rows[0]["state_name"] == "JAMMU & KASHMIR"


def wiggly(n=200):
    """A square whose right edge zig-zags by 0.001 degrees: many vertices, nearly the same shape."""
    right = [[1 + (0.001 if i % 2 else 0), i / n] for i in range(n + 1)]
    return {"type": "Polygon", "coordinates": [[[0, 0], *right, [0, 1], [0, 0]]]}


def test_simplify_writes_only_the_four_join_properties_and_returns_the_byte_size(tmp_path):
    new = load_new_from(new_fc((502, "Anantapur", 28, "ANDHRA PRADESH", wiggly())))
    out = tmp_path / "s.geojson"
    size = simplify(new, 0.005, out)
    assert size == out.stat().st_size > 0
    fc = json.loads(out.read_text(encoding="utf-8"))
    assert fc["type"] == "FeatureCollection" and len(fc["features"]) == 1
    f = fc["features"][0]
    assert f["properties"] == {"dist_lgd": 502, "dtname": "Anantapur", "state_lgd": 28, "stname": "ANDHRA PRADESH"}
    geom = shape(f["geometry"])
    assert geom.is_valid and len(geom.exterior.coords) < 20 < len(new[0]["geom"].exterior.coords)
    assert abs(geom.area - 1.0) < 0.01                                           # simplified, not distorted


def test_simplify_leaves_a_feature_with_no_area_out_of_the_map_file(tmp_path):
    new = load_new_from(new_fc((502, "Anantapur", 28, "ANDHRA PRADESH", square(0, 0, 1, 1)),
                               (999, "Flat", 28, "ANDHRA PRADESH", collinear_ring())))
    out = tmp_path / "s.geojson"
    simplify(new, 0.005, out)
    fc = json.loads(out.read_text(encoding="utf-8"))
    assert [f["properties"]["dist_lgd"] for f in fc["features"]] == [502]
    assert all(f["geometry"]["type"] in ("Polygon", "MultiPolygon") for f in fc["features"])


def test_simplify_within_retries_coarser_tolerances_until_the_file_fits(tmp_path):
    new = load_new_from(new_fc((502, "Anantapur", 28, "ANDHRA PRADESH", wiggly())))
    out = tmp_path / "s.geojson"
    assert simplify_within(new, out, tolerances=(0.0001, 0.001, 0.01), limit_bytes=10 ** 9) == (0.0001, out.stat().st_size)
    tolerance, size = simplify_within(new, out, tolerances=(0.0001, 0.001, 0.01), limit_bytes=0)
    assert tolerance == 0.01 and size == out.stat().st_size                     # the last one is left on disk
