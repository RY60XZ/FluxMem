"""Errors raised at application use-case boundaries."""


class MessageNotFoundError(LookupError):
    """The selected evidence message does not exist in the requested scope."""


class SessionNotFoundError(LookupError):
    """The selected session does not exist in the requested user scope."""


class InvalidMemoryScopeError(ValueError):
    """A memory was assigned a session scope outside its evidence session."""


class InvalidRetrievalContextError(ValueError):
    """Messages supplied to retrieval do not describe the same session."""
