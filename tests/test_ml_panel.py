"""The unit table: members summed, the state from the largest member, estimated flows across the unit's boundary
from the pieces log, and neighbours from the map."""
import json

import pandas as pd

from agents.schemas import CORE_CATEGORIES
from ml import panel


def rebased(rows):
    """(district_code, state, year, total) -> re-based district rows with the total in the first category."""
    out = []
    for code, state, year, total in rows:
        cats = {c: 0.0 for c in CORE_CATEGORIES}
        cats[CORE_CATEGORIES[0]] = float(total)
        out.append({"unit_id": f"d{code}", "state_code": state, "district_code": code, "year": year, **cats, "total": float(total),
                    "is_estimated": 0, "share": 1.0, "share_basis": "observed", "holder_code": None})
    df = pd.DataFrame(out)
    df["district_code"] = df["district_code"].astype("Int64")
    return df


UNITS = {"u1": [1, 2], "u2": [3]}


def test_build_panel_sums_members_and_counts_estimates_that_cross_the_boundary():
    df = rebased([(1, 9, 2001, 30), (2, 9, 2001, 70), (1, 9, 2002, 40), (2, 9, 2002, 60), (3, 10, 2001, 50), (3, 10, 2002, 55)])
    pieces = pd.DataFrame([{"member_code": 2, "holder_code": 1, "kind": "split", "year": 2001, "amount": 20.0},     # inside u1: exact
                           {"member_code": 2, "holder_code": 3, "kind": "split", "year": 2001, "amount": 5.0}])     # from u2 into u1
    out = panel.build_panel(df, UNITS, pieces)
    assert list(out.columns) == panel.COLUMNS and len(out) == 4
    u1 = out[out["unit_id"] == "u1"].set_index("year")
    assert u1.loc[2001, "total"] == 100 and u1.loc[2002, "total"] == 100 and u1.loc[2001, "n_members"] == 2 and u1.loc[2001, "state_code"] == 9
    assert u1.loc[2001, "outside_est"] == 5.0 and u1.loc[2001, "outside_share"] == 0.05 and u1.loc[2002, "outside_est"] == 0.0
    u2 = out[out["unit_id"] == "u2"].set_index("year")
    assert u2.loc[2001, "outside_est"] == 5.0 and u2.loc[2001, "outside_share"] == 0.1          # the same piece left u2
    assert panel.build_panel(df, UNITS, None)["outside_est"].eq(0).all()


def square(x0, y0, size=1.0):
    return {"type": "Polygon", "coordinates": [[[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size], [x0, y0]]]}


def test_neighbours_come_from_touching_polygons_and_never_include_the_unit_itself(tmp_path):
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"dist_lgd": 1, "dtname": "A"}, "geometry": square(0, 0)},
        {"type": "Feature", "properties": {"dist_lgd": 2, "dtname": "B"}, "geometry": square(1, 0)},        # shares an edge with A
        {"type": "Feature", "properties": {"dist_lgd": 3, "dtname": "C"}, "geometry": square(5, 5)},        # far away
        {"type": "Feature", "properties": {"dist_lgd": 0, "dtname": "unassigned"}, "geometry": square(0, 1)}]}
    p = tmp_path / "map.geojson"
    p.write_text(json.dumps(fc), encoding="utf-8")
    dn = panel.district_neighbours(p)
    assert dn == {1: {2}, 2: {1}}
    assert panel.unit_neighbours({"u1": [1], "u2": [2], "u3": [3]}, dn) == {"u1": ["u2"], "u2": ["u1"], "u3": []}
    assert panel.unit_neighbours({"u12": [1, 2], "u3": [3]}, dn) == {"u12": [], "u3": []}                    # A and B in one unit
