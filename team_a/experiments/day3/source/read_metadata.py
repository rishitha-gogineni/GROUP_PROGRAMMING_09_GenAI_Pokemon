import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, sys.argv[1])
destination=Path(sys.argv[2])
result={"status":"not_loaded","attack_damage":{},"card_hp":{}}
try:
    from cg.api import all_attack, all_card_data
    damage={int(item.attackId):int(item.damage or 0) for item in all_attack()}
    hp={int(item.cardId):int(item.hp or 0) for item in all_card_data()}
    result.update(status="loaded_from_official_sdk",attack_damage=damage,card_hp=hp)
except Exception as error:
    result.update(status="unavailable",error_type=type(error).__name__,error_message=str(error),traceback=traceback.format_exc())
destination.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
