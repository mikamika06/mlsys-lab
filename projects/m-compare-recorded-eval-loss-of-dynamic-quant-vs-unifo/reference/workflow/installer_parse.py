import re

def parse_installer_transcript(transcript):
  pattern = r"Successfully installed\s+([a-zA-Z0-9_\-]+)-([0-9\.\+a-zA-Z]+)"
  matches = re.findall(pattern, transcript)
  return dict(matches)
