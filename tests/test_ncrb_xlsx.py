"""NCRB's district workbook: state markers, total rows, the header's merged gaps, and the Excel row number."""
import openpyxl
import pytest

from data_ingestion.ncrb_xlsx import crime_columns, frame, read_district_rows


@pytest.fixture
def workbook(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "District Data Report"
    ws.append(["District-wise Crimes committed against Women during 2015"])
    ws.append(["SL", "District", "YEAR", "Rape", None, "Rape_Gang Rape", "Attempt to commit Rape"])   # a merged gap
    ws.append(["State: Andhra Pradesh"])
    ws.append([1, "Anantapur", 2015, 40, None, 2, 3])
    ws.append([2, "Guntakal Railway", 2015, 1, None, 0, 0])
    ws.append([3, "Total District(s)", None, 41, None, 2, 3])
    ws.append(["UT: Delhi UT"])
    ws.append([1, "Metro", 2015, 7.0, None, None, 1])
    ws.append([2, "Total District(s)", None, 7, None, 0, 1])
    ws.append(["Source: Crime in India"])
    path = tmp_path / "w.xlsx"
    wb.save(path)
    return path


def test_rows_carry_their_state_their_excel_row_and_every_head_as_text(workbook):
    rows = read_district_rows(workbook)
    assert [(r["state_name"], r["district"], r["source_row"]) for r in rows] == [
        ("Andhra Pradesh", "Anantapur", 4), ("Andhra Pradesh", "Guntakal Railway", 5), ("Delhi UT", "Metro", 8)]
    assert rows[0] == {"state_name": "Andhra Pradesh", "district": "Anantapur", "source_row": 4, "YEAR": "2015",
                       "Rape": "40", "Rape_Gang Rape": "2", "Attempt to commit Rape": "3"}
    assert (rows[2]["Rape"], rows[2]["Rape_Gang Rape"]) == ("7", "")              # 7.0 -> '7'; an empty cell stays empty


def test_crime_columns_skip_the_key_labels_and_the_merged_gaps(workbook):
    assert crime_columns(workbook) == ["Rape", "Rape_Gang Rape", "Attempt to commit Rape"]


def test_frame_is_numeric_with_an_integer_year(workbook):
    df = frame(workbook)
    assert list(df["year"]) == [2015, 2015, 2015]
    assert df["Rape"].tolist() == [40, 1, 7] and df["Rape_Gang Rape"].isna().tolist() == [False, False, True]


def test_a_district_before_any_state_marker_is_an_error(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["SL", "District", "YEAR", "Rape"])
    ws.append([1, "Nowhere", 2015, 1])
    wb.save(tmp_path / "bad.xlsx")
    with pytest.raises(ValueError, match="before any 'State:' marker"):
        read_district_rows(tmp_path / "bad.xlsx")
