from typing import Annotated

from pydantic import StringConstraints

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]
