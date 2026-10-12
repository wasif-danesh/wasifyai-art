import io, json, sys, time, urllib.request, os
from PIL import Image
B="http://localhost:8010"; VID=open(sys.argv[1]).read().strip(); DATA=sys.argv[2]
def call(m,p,data=None,raw=None,ctype=None):
    h={"content-type": ctype or "application/json"}
    body = raw if raw is not None else (json.dumps(data).encode() if data is not None else None)
    try:
        d=urllib.request.urlopen(urllib.request.Request(B+p, body, h, method=m), timeout=60).read()
        return json.loads(d) if d else {}
    except urllib.error.HTTPError as e:
        return {"_status": e.code, **json.loads(e.read() or b"{}")}
def wait(jid, until=("succeeded","failed","canceled"), cap=240):
    t=time.time()
    while time.time()-t<cap:
        s=call("GET",f"/v1/jobs/{jid}")
        if s["status"] in until: return s
        time.sleep(1)
    raise SystemExit("timeout "+jid)
base={"provider_id":"comfyui-local","version_id":VID}
# 1. a real "photo" with EXIF orientation, uploaded then edited
im=Image.open(sys.argv[3]).convert("RGB"); ex=Image.Exif(); ex[0x0112]=1; ex[0x010F]="SecretCameraMaker"
buf=io.BytesIO(); im.save(buf,"JPEG",quality=92,exif=ex)
up=call("POST","/v1/uploads/images",raw=buf.getvalue(),ctype="image/jpeg")
print("upload:", up)
files=lambda: sorted(p for p,_,fs in os.walk(DATA) for f in fs for p in [os.path.join(p,f)])
print("files after upload:", [f.split("/local/wasifyai-art/")[-1] for f in files()])
j=call("POST","/v1/generations",{**base,"prompt":"Turn this into a soft watercolor painting","mode":"edit","source_upload_id":up["upload_id"]})
t=time.time(); s=wait(j["id"]); print(f"edit from upload: {s['status']} in {time.time()-t:.0f}s", s.get("error") or "")
print("files after edit:", [f.split("/local/wasifyai-art/")[-1] for f in files()])
c=call("GET",f"/v1/creations/{s['creation_id']}"); print("creation:", c["width"],"x",c["height"], c["params"]["mode"], c["params"]["source"], "parent", c["parent_id"])
open(sys.argv[4]+"/watercolor.png","wb").write(urllib.request.urlopen(B+c["image_url"]).read())
# 2. cancel a running job
j=call("POST","/v1/generations",{**base,"prompt":"A castle on a hill at sunset, painterly","size":"wide","seed":99})
wait(j["id"], until=("running",)); time.sleep(3)
r=call("POST",f"/v1/jobs/{j['id']}/cancel"); print("cancel request ->", r["status"])
t=time.time(); s=wait(j["id"]); print(f"after cancel: {s['status']} (stopped in {time.time()-t:.1f}s)", s.get("creation_id"))
# 3. retry, and ComfyUI is healthy afterwards
r=call("POST",f"/v1/jobs/{j['id']}/retry"); print("retry:", r["status"], r["id"]!=j["id"])
t=time.time(); s=wait(r["id"]); print(f"retried job: {s['status']} in {time.time()-t:.0f}s")
print("jobs listed:", [(x["status"], x["prompt"][:22]) for x in call("GET","/v1/jobs")["jobs"]])
