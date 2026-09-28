from pathlib import Path

from whats_for_dinner.recipes import load_recipes


def _write_recipe(tmp_path: Path, filename: str, content: str) -> None:
    (tmp_path / filename).write_text(content, encoding="utf-8")


def test_load_recipes_extracts_title_and_source(tmp_path: Path) -> None:
    _write_recipe(tmp_path, "01.txt", "Quick Chicken Stir-Fry\n\nIngredients:\n    chicken\n")

    documents = load_recipes(tmp_path)

    assert len(documents) == 1
    assert documents[0].meta["title"] == "Quick Chicken Stir-Fry"
    assert documents[0].meta["source"] == "01.txt"
    assert "chicken" in (documents[0].content or "")


def test_load_recipes_title_skips_leading_blank_lines(tmp_path: Path) -> None:
    _write_recipe(tmp_path, "03.txt", "\n\nSimple Guacamole\n\nIngredients:\n    avocado\n")

    documents = load_recipes(tmp_path)

    assert documents[0].meta["title"] == "Simple Guacamole"


def test_load_recipes_skips_empty_files(tmp_path: Path) -> None:
    _write_recipe(tmp_path, "empty.txt", "   \n  \n")
    _write_recipe(tmp_path, "01.txt", "Real Recipe\ncontent")

    documents = load_recipes(tmp_path)

    assert len(documents) == 1
    assert documents[0].meta["source"] == "01.txt"


def test_same_filename_and_content_produce_same_id(tmp_path: Path) -> None:
    content = "Quick Chicken Stir-Fry\n\nIngredients:\n    chicken\n"
    _write_recipe(tmp_path, "01.txt", content)
    first_pass = load_recipes(tmp_path)

    _write_recipe(tmp_path, "01.txt", content)  # simulate a second startup, same file
    second_pass = load_recipes(tmp_path)

    assert first_pass[0].id == second_pass[0].id


def test_editing_content_changes_the_document_id(tmp_path: Path) -> None:
    _write_recipe(tmp_path, "01.txt", "Quick Chicken Stir-Fry\n\nIngredients:\n    chicken\n")
    before = load_recipes(tmp_path)[0].id

    _write_recipe(
        tmp_path, "01.txt", "Quick Chicken Stir-Fry\n\nIngredients:\n    chicken, extra garlic\n"
    )
    after = load_recipes(tmp_path)[0].id

    assert before != after


def test_different_filenames_with_identical_content_produce_different_ids(tmp_path: Path) -> None:
    content = "Identical Recipe\n\nsame text"
    _write_recipe(tmp_path, "01.txt", content)
    _write_recipe(tmp_path, "02.txt", content)

    documents = load_recipes(tmp_path)

    assert documents[0].id != documents[1].id


def test_trivial_whitespace_differences_do_not_change_the_id(tmp_path: Path) -> None:
    _write_recipe(tmp_path, "01.txt", "Title\n\nIngredients:\n    chicken\n    rice\n")
    first_id = load_recipes(tmp_path)[0].id

    _write_recipe(
        tmp_path, "01.txt", "Title\n\nIngredients:\n  chicken\n  rice\n"
    )  # reformatted, same words
    second_id = load_recipes(tmp_path)[0].id

    assert first_id == second_id
