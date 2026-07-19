def check_bureau_score(pan: str | None, name: str | None = None, dob: str | None = None, mobile: str | None = None) -> int:
    """
    Mock Direct Bureau Integration.
    Returns:
      -1 if No Hit (New To Credit)
      > 0 (e.g., 745) if traditional credit history exists.
    """
    if not pan:
        return -1
        
    last_char = pan[-1].upper()
    if 'A' <= last_char <= 'M':
        return -1  # NTC
    elif 'N' <= last_char <= 'Z':
        return 745  # Traditional Hit
        
    return -1
