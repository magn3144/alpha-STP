import site
from pathlib import Path


repository = Path(__file__).resolve().parents[1]
alphaproof = repository.parent / "AlphaProof"
site_packages = Path(site.getsitepackages()[0])
link = site_packages / "alphaproof-editable.pth"
link.write_text(str(alphaproof.resolve()) + "\n", encoding="utf-8")
print(f"Linked {alphaproof} through {link}")
