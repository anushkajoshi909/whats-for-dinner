# Python Project Conventions

## Core Principles

- Use modern Python (3.12+) with strict type hints
- Make interfaces clear and explicit
- Prioritize readability and maintainability
- Separate business logic flows from implementation details
- Use async where it makes sense
- Avoid `# type: ignore` and `Any`
- Use Generics where appropriate

## Function Categories

1. Business Logic Functions

   - Readable, top-down flow
   - Minimal implementation details
   - Clear conditional paths

2. Helper Functions
   - Functional and stateless
   - Single purpose
   - Implementation focused

Note: Split functions that serve multiple purposes based on conditions

## Development Environment

- Package and project management: `uv`
- Code formatting/linting/static analysis: ruff and pyright
- Testing: pytest with colocated tests

## Type System Conventions

```python
# Use built-in types
list, dict, type  # Not List, Dict, Type

# Use pipe operator for unions
str | int  # Not Union[str, int]
str | None  # Not Optional[str]

# Prefer Pydantic models, use TypedDicts when necessary
# Avoid dataclasses
```

## Code Structure Patterns

### Early Returns

Prefer early returns to reduce nesting and improve readability. Use them for:

- Guard clauses (validation, preconditions)
- Special cases or alternative paths
- Error conditions

```python
def process_user(user: User) -> None:
    if not user.is_active:
        return

    # Main flow continues without nesting
    process_active_user(user)
```

### API Layer (FastAPI)

```python
@router.post("/orders")
async def create_order(order: OrderCreate) -> OrderResponse:
    # Thin controllers, delegate to service layer
    return await order_service.create_order(order)
```

### Service Layer

```python
async def process_order(order: OrderCreate) -> Order:
    # Business logic flows should be readable top-down
    validated_order = await validate_order_items(order)
    payment = await process_payment(validated_order)
    return await create_order_with_payment(validated_order, payment)
```

### Repository Pattern

```python
class UserRepository:
    # Encapsulate database operations
    async def get_by_email(self, email: str) -> User | None:
        return await User.get_by_email(email)
```

## Database

- Use SQLModel for database models (based on Pydantic and SQLAlchemy)
- No manual alembic migrations - use `alembic --autogenerate`
- Keep database logic isolated in repositories

## Testing

- Colocate tests in `<module>_test.py`
- Mark async tests with `@pytest.mark.anyio`
- Use async fixtures with `async def`
- Use `async def` for test functions that rely on async fixtures

```python
async def test_create_user(client: AsyncClient):
    response = await client.post("/users", json={"name": "Test"})
    assert response.status_code == 200
```

## Error Handling

```python
# Define domain-specific exceptions
class OrderError(Exception): pass
class InsufficientInventoryError(OrderError): pass

# Handle at API layer
@app.exception_handler(OrderError)
async def order_error_handler(request, exc: OrderError):
    return JSONResponse(status_code=400, content={"error": str(exc)})
```

## Logging

```python
import logging
logger = logging.getLogger(__name__)

# Use for debugging
logger.info("Processing order", extra={"order_id": order.id})
```

## Documentation

### Docstrings

- Add detailed docstrings (args, returns, raises) only for:
  - Public API endpoints
  - Service layer entry points
  - Class constructors
  - Complex public functions that other developers will use
- For internal/helper functions, focus on WHY when documentation is needed

```python
# Detailed docstring for entry point
async def create_order(order_data: OrderCreate) -> Order:
    """Create a new order in the system.

    Args:
        order_data: Order creation data including items and customer info

    Returns:
        Created order with generated ID and status

    Raises:
        InvalidOrderError: If order validation fails
        PaymentError: If payment processing fails
    """

# Simple or no docstring for internal helper
async def validate_order_items(items: list[OrderItem]) -> bool:
    # Basic validation before order processing
    return all(item.quantity > 0 for item in items)
```

### Code Comments

- Explain WHY, not WHAT
- Focus on:
  - Business context and domain knowledge
  - Non-obvious implications
  - Complex business rules
  - Workarounds and their reasons
  - Links to relevant documentation or tickets
  - Warning about potential pitfalls

Bad:

```python
# open the file
f = open(file_path)
```

Good:

```python
# Cache must be warmed up before serving requests
# to ensure < 100ms response times
warm_up_cache()
```

If you need to explain WHAT the code does, consider:

- Improving variable/function names
- Breaking complex logic into well-named helper functions
- Adding type hints
- Using more explicit data structures

## Configuration

- Use Pydantic Settings with environment variables
- Separate configurations for different environments

# Git

## Commit Messages

We follow a simplified version of [Conventional Commits](https://www.conventionalcommits.org/). Each commit message must follow the format:

`<type>[!]: <description>`

Where `description` is a clear, concise summary of the change, and `type` must be one of:

- `feat`: New features or significant enhancements
- `fix`: Bug fixes and error corrections
- `chore`: Maintenance tasks, dependency updates, docs, test coverage, etc.
- `refactor`: Code restructuring without adding features or fixing bugs

For breaking changes, append a `!` after the type. Examples:
```
feat: implement user authentication flow
fix: handle empty API responses
feat!: change authentication API interface
```

Note: We use a simplified format that does not include scopes (e.g., no parentheses) or footers. Breaking changes are indicated using the `!` notation.

## Branch Naming

Branch names should follow the pattern:
`<type>-<description>`

Where `type` matches our commit types and `description` is a brief, hyphen-separated description:

- `feat-user-authentication`
- `fix-api-response-handling`
- `chore-dependency-updates`


Optionally, include the ticket number after the type and before the description:

- `feat-CON-123-user-authentication`

## Commit Signing

We recommend signing commits with your SSH or GPG key, see
- https://docs.github.com/en/authentication/managing-commit-signature-verification/signing-commits and
- https://docs.github.com/en/authentication/managing-commit-signature-verification/telling-git-about-your-signing-key#telling-git-about-your-ssh-key for more information.

When creating commits, use the `-s` flag to sign the commit, for example:

```
git commit -s -m "feat: implement user authentication flow"
```
