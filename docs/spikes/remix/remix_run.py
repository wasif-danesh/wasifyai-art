import io, json, sys, time, urllib.request
from PIL import Image, ImageChops
B="http://localhost:8010"
VID=open(sys.argv[1]).read().strip()
def post(path, data):
    r=urllib.request.Request(B+path, json.dumps(data).encode(), {"content-type":"application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=30).read())
def get(path, raw=False):
    d=urllib.request.urlopen(B+path, timeout=60).read()
    return d if raw else json.loads(d)
def run(body, label):
    t=time.time(); j=post("/v1/generations", body)
    while True:
        s=get(f"/v1/jobs/{j['id']}")
        if s["status"] in ("succeeded","failed"): break
        time.sleep(2)
    print(f"{label}: {s['status']} in {time.time()-t:.0f}s", s.get("error") or "")
    return get(f"/v1/creations/{s['creation_id']}") if s["creation_id"] else None
base={"provider_id":"comfyui-local","version_id":VID}
a=run({**base,"prompt":"A red fox in fresh snow at dawn, soft golden light, photographic","size":"landscape","seed":7},"original")
b=run({**base,"prompt":a["prompt"],"size":"landscape","seed":a["params"]["seed"],"parent_id":a["id"]},"remix settings (same seed)")
ia=Image.open(io.BytesIO(get(a["image_url"],True))).convert("RGB"); ib=Image.open(io.BytesIO(get(b["image_url"],True))).convert("RGB")
diff=ImageChops.difference(ia,ib); bbox=diff.getbbox(); mx=max(max(x) for x in diff.getextrema())
print("same size:", ia.size==ib.size, "| identical:", bbox is None, "| max pixel diff:", mx)
print("parent link ok:", b["parent_id"]==a["id"])
c=run({**base,"prompt":"Change the scene to dusk, with warm glowing lights in the snow behind the fox","mode":"edit","source_creation_id":a["id"]},"remix picture (edit)")
print("edit: parent ok:", c["parent_id"]==a["id"], c["width"],"x",c["height"], "mode", c["params"]["mode"], "steps", c["params"]["steps"])
for n,x in (("orig",a),("remix",b),("edit",c)):
    open(f"{sys.argv[2]}/{n}.png","wb").write(get(x["image_url"],True))
json.dump({"a":a["id"],"b":b["id"],"c":c["id"]}, open(sys.argv[2]+"/ids.json","w"))
