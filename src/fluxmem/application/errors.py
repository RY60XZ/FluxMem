"""Errors raised at application use-case boundaries."""


class MessageNotFoundError(LookupError):
    """The selected evidence message does not exist in the requested scope."""
