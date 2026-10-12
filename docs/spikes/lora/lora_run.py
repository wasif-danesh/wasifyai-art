import io, json, sys, time, urllib.request
from PIL import Image, ImageChops, ImageStat
B="http://localhost:8010"; OUT=sys.argv[1]
def call(method, path, data=None, raw=False):
    r=urllib.request.Request(B+path, json.dumps(data).encode() if data is not None else None, {"content-type":"application/json"}, method=method)
    d=urllib.request.urlopen(r, timeout=60).read()
    return d if raw else json.loads(d)
models=call("GET","/v1/models")["models"]
ver={m["slug"]:m["versions"][0]["id"] for m in models}
base=ver["flux-2-klein-4b"]; lora=ver["synthetic-noise-lora"]
call("PATCH",f"/v1/model-versions/{base}",{"family":"flux2-klein","status":"ready"})
call("PATCH",f"/v1/model-versions/{lora}",{"family":"flux2-klein","trigger_words":["synthx"],"status":"ready"})
def run(label, **extra):
    t=time.time()
    j=call("POST","/v1/generations",{"provider_id":"comfyui-local","version_id":base,"prompt":"A red fox in fresh snow at dawn, soft golden light, photographic","size":"square","seed":7,**extra})
    while True:
        s=call("GET",f"/v1/jobs/{j['id']}")
        if s["status"] in ("succeeded","failed"): break
        time.sleep(2)
    print(f"{label:28} {s['status']} {time.time()-t:3.0f}s", s.get("error") or "")
    c=call("GET",f"/v1/creations/{s['creation_id']}")
    img=Image.open(io.BytesIO(call("GET",c["image_url"],raw=True))).convert("RGB")
    img.save(f"{OUT}/{label.replace(' ','_')}.png")
    return c,img
def diff(a,b):
    d=ImageChops.difference(a,b); m=ImageStat.Stat(d).mean
    return f"mean abs diff {sum(m)/3:5.2f}, identical={d.getbbox() is None}"
c0,i0=run("no lora")
c1,i1=run("lora 1.0", loras=[{"version_id":lora,"weight":1.0}])
c2,i2=run("lora 0.0", loras=[{"version_id":lora,"weight":0.0}])
c3,i3=run("lora 1.0 again", loras=[{"version_id":lora,"weight":1.0}])
print("no lora  vs lora 1.0  :", diff(i0,i1))
print("no lora  vs lora 0.0  :", diff(i0,i2))
print("lora 1.0 vs lora 1.0  :", diff(i1,i3))
print("models recorded:", [(m["role"],m["name"],m["weight"]) for m in c1["models"]])
j=call("POST","/v1/generations",{"provider_id":"comfyui-local","version_id":base,"prompt":"Change the scene to dusk","mode":"edit","source_creation_id":c0["id"],"loras":[{"version_id":lora,"weight":0.6}]})
while True:
    s=call("GET",f"/v1/jobs/{j['id']}")
    if s["status"] in ("succeeded","failed"): break
    time.sleep(2)
print("edit with lora:", s["status"], s.get("error") or "")
json.dump({"plain":c0["id"],"lora":c1["id"],"edit":s["creation_id"]}, open(f"{OUT}/ids.json","w"))
