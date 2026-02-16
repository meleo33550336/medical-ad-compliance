from datetime import datetime, timezone
import sys
from pathlib import Path

# ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.database import save_report, get_reports

report = {
    'timestamp': datetime.now(timezone.utc).isoformat(),
    'input_text': '测试保存报告 - This is a test report.',
    'tokens': ['测试','保存','报告'],
    'regex_matches': [],
    'violation_rule_matches': [],
    'semantic_matches': [],
    'classifier': None,
    'verdict': '合规'
}

save_report(report)
rows = get_reports(5)
if rows:
    latest = rows[0]
    print(f"Saved report id={latest['id']}, timestamp={latest['timestamp']}, verdict={latest['verdict']}")
else:
    print('No reports found')
