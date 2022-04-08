from dataclasses import dataclass
import json
import os


import pandas as pd

class GameLogger:
    def __init__(self, logfile: str = 'logs/compete/compete.log'):
        self.logfile = logfile
        
    def log(self, outcome: dict) -> None:
        with open(self.logfile, 'a') as f:
            json.dump(outcome, f)
            f.write('\n')
            
@dataclass
class LogReader:
    logpath: str
    
    def list_h2h(self):
        with open(self.logpath, 'r') as f:
            h2h = [json.loads(line)['h2h'] for line in f]
        return pd.Series(h2h).value_counts()
                
    def get_records_by_h2h(self, h2h: str = None):
        records = pd.read_json(self.logpath, lines=True)
        if h2h:
            records = records[records['h2h']==h2h]
        return records
    
    def clear_logs(self):
        resp = input("""you are about to erase all the logs with inter-agent game records. 
        is it really what you want? (y/n)""")
        if resp == 'y':
            with open(self.logpath, 'w') as f:
                f.write('')
                print(f'{os.path.abspath(self.logpath)} has been cleared.')
        
            