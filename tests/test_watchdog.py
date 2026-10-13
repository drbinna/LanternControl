import sys, types, importlib.util
# --- stub ROS modules
class Node:
    sites={}
    def __init__(self,name): self.logs=[]
    def create_publisher(self,*a): 
        p=types.SimpleNamespace(msgs=[]); p.publish=p.msgs.append; return p
    def create_subscription(self,*a): pass
    def create_timer(self,*a): pass
    def get_logger(self):
        L=self.logs; import inspect
        sites=Node.sites
        def mk(sev):
            def f(msg):
                fr=inspect.stack()[1]; key=(fr.filename,fr.lineno)
                if sites.setdefault(key,sev)!=sev: raise ValueError('Logger severity cannot be changed between calls.')
                L.append((sev,msg))
            return f
        return types.SimpleNamespace(info=mk('I'),warn=mk('W'),error=mk('E'))
    def get_clock(self): return types.SimpleNamespace(now=lambda: types.SimpleNamespace(nanoseconds=0))
def mod(n,**k):
    m=types.ModuleType(n); m.__dict__.update(k); sys.modules[n]=m
mod('rclpy'); mod('rclpy.node',Node=Node)
mod('rclpy.qos',QoSProfile=lambda **k:None,ReliabilityPolicy=types.SimpleNamespace(BEST_EFFORT=0),DurabilityPolicy=types.SimpleNamespace(VOLATILE=0,TRANSIENT_LOCAL=1),HistoryPolicy=types.SimpleNamespace(KEEP_LAST=0))
mod('nav_msgs'); mod('nav_msgs.msg',Odometry=object)
mod('std_msgs'); mod('std_msgs.msg',String=lambda: types.SimpleNamespace(data=''))
class VO: pass
mod('px4_msgs'); mod('px4_msgs.msg',VehicleOdometry=VO)
path=sys.argv[1] if len(sys.argv)>1 else __import__('os').path.join(__import__('os').path.dirname(__file__),'..','vision','vslam_to_px4.py')
spec=importlib.util.spec_from_file_location('b',path); b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
br=b.Bridge()
def odom(t,x,y,z,vx=0.0):
    N=types.SimpleNamespace
    return N(header=N(stamp=N(sec=int(t),nanosec=int((t%1)*1e9))),
      pose=N(pose=N(position=N(x=x,y=y,z=z),orientation=N(w=1,x=0,y=0,z=0)),covariance=[0.0001]*36),
      twist=N(twist=N(linear=N(x=vx,y=0,z=0))))
t=100.0; x=0.0
for i in range(300):            # 3.3 s smooth walk at 0.5 m/s
    t+=1/90; x+=0.5/90; br.cb(odom(t,x,0,0,0.5))
print('after smooth:',br.state,'sent',len(br.pub.msgs),'rejected',br.rejected)
t+=1/90; br.cb(odom(t,x+8,0,0))  # 8 m teleport
print('after jump:',br.state,br.reason)
for i in range(60):
    t+=1/90; br.cb(odom(t,x+8,0,0))
print('after recovery:',br.state,'reset_counter',br.reset_counter,'last x sent',br.pub.msgs[-1].position[0])
print([l for l in br.logs if l[0]!='I'])

# second loss/recovery cycle exercises every log site again
for i in range(3):
    t+=1/90; br.cb(odom(t,x+20+i*10,0,0))
    for j in range(60): t+=1/90; br.cb(odom(t,x+20+i*10,0,0))
print('cycles done:',br.state,'losses',br.losses,'reset_counter',br.reset_counter)
