# Lessons Learned

## Duplicate method definitions (Python)
**Pattern**: When adding a new method to a large class, always search the full
file for an existing method with the same name before inserting. Python silently
takes the last definition — adding a correct method early in the class is a
no-op if a stale version exists later.
**Fix**: `Grep` the file for `def <method_name>` before writing.

## Static methods that acquire loggers
**Pattern**: `@staticmethod` methods that call `logging.getLogger("ClassName")`
create an invisible coupling and bypass the instance logger. If the class logger
is configured differently (level, handlers), the static method will log to the
wrong place.
**Fix**: Convert to an instance method and use `self.logger`.

## Requirements.txt deleted — use environment.yml
**Context**: `requirements.txt` was deleted from this project. The install
method is `conda env create -f environment.yml`. Do NOT reference requirements.txt
in docs or instructions.

## Artifact cleanup must be coordinated
**Pattern**: Storing a model in `self._some_model` and then referencing a
*different* attribute name in `cleanup()` makes cleanup a silent no-op.
**Fix**: Always grep for all attribute assignments before writing cleanup logic.

## Whisper model must be stored on self
**Pattern**: `_transcribe` stored the model in a local variable `model` but
`cleanup()` checked `self._whisper_model`. No memory was ever freed.
**Fix**: Always assign to `self._attr` when the object must survive a method call.
