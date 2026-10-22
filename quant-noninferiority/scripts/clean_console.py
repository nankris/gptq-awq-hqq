import re,sys
s=open(sys.argv[1]).read()
s=re.sub(r'\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\)|[=>])','',s)
s='\n'.join(x for x in s.splitlines() if x.strip())
print(s[-10000:])
