import redis
import pickle

class Redis(object):
    def __init__(self):
        self.redis = None

    def connect(self,host="127.0.0.1",port=6379,db=0):
        self.redis = redis.Redis(host=host,port=port ,db=db)

    def setRedis(self, name, model):
        self.redis.set(name, pickle.dumps(model))

    def getRedis(self, name):
        obj = self.redis.get(name)
        return pickle.loads(obj)

    def delRedis(self,name):
        self.redis.delete(name)