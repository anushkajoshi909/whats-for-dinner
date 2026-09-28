# GenAI Pipeline Challenge

You are tasked with creating a small Python application that provides recipe recommendations based on the supplied list of ingredients.

We are looking for a PoC style solution that is well thought out, well documented, and well structured. We are not looking for a perfect solution, but rather a solution that you can explain, defend, and is heading in the right direction. Spend no more than 5 hours on this challenge, avoid the temptation to over-engineer or polish it.

## Requirements

### 1. API Endpoint

- **Endpoint:** `/recommend_recipe`
- **Framework:** FastAPI
- **Functionality:**
  - Accepts ingredients as text input.
  - Uses ChatGPT 4o (API key will be provided).
  - Flexible text input (no specific schema required).
  - Returns recipe instructions as a text string, ideally Markdown formatted.
  - **Bonus:** Accepts an image of the ingredients along with the text. This is a bit tricky so we have also provided a helper custom component for inspiration, if you get this far.

### 2. Data Sources

- **Recipes:** Use the provided dataset of recipes in `data.zip`.
- **Database:** Leverage a Postgres database to find the best matching recipes.
  - Docker-compose file with an empty Postgres database is provided.
  - Handle data ingestion during startup or as a secondary command.
  - **Preferred Solution:** Use [RAG](https://haystack.deepset.ai/blog/rag-pipelines-from-scratch) with ChatGPT to generate recipe instructions.
  - **Recommendation:** Use [Haystack 2.x](https://haystack.deepset.ai/).

### 3. Project Setup

- **Configuration:** Provided `pyproject.toml` using [uv](https://docs.astral.sh/uv/).
- **Project Layout:** Use a [`src` layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/).

### 4. Documentation

- **README:** Include a brief README explaining:
  - How to run the application.
  - Solution architecture.
  - Assumptions made.
  - Examples of how to use the API.

### 5. Time Management

- **Research:** Spend the first hour researching relevant FastAPI and Haystack tutorials.
- **Development Time:** Spend no more than 5 hours on this challenge.

## Additional Notes

- **Pillow Dependency:**
  - Additional dependencies may be needed. See [Pillow documentation](https://pillow.readthedocs.io/en/latest/installation/building-from-source.html#external-libraries).
  - If issues persist, message us briefly and remove the dependency to unblock yourself.
- **psycopg-binary Dependency:**
  - Refer to their [documentation](https://www.psycopg.org/docs/install.html#build-prerequisites) if you encounter issues.

## Submission
- When finished, email us a zip archive of your `git` repository.
    -  Use `git archive --format=zip HEAD -o "data-eng-submission-$NAME.zip"`
- Ensure the repository contains all source code, the `README.md`, dbt project files, and any other files required to run the solution.

> [!NOTE]
> A final note/suggestion. We know that this is not a trivial task, beyond the technical details in the code, we are also looking to see how you approach it and what decisions you make. Don't be afraid to make assumptions or adjust on the fly, but be prepared to explain why you made those decisions.
>
> Good luck!
