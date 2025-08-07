import numpy as np
import os.path as op
import statistics


class LineCreator():
    def __init__(self,length = 400):
        super(LineCreator, self).__init__()
        self.length = length

    def Reverse(self,x):
        return np.flipud(x)

    def AddNoise(self,x,scale=1000):
        A = (np.max(x) - np.min(x)) / scale
        noise = np.random.randn(len(x)) * A
        return x + noise

    def Move(self,x,x1,x2,x3):
        assert x3 < x1 or x3 > x2
        temp = list()
        partial = x[x1:x2]
        if x3 < x1:
            temp.append(x[:x3])
            temp.append(partial)
            temp.append(x[x3:x1])
            temp.append(x[x2:])
        if x3 > x2:
            temp.append(x[:x1])
            temp.append(x[x2:x3])
            temp.append(partial)
            temp.append(x[x3:])
        return np.concatenate(temp,0)

    def Change(self,x,x1,x2,x3,x4):
        temp = list()
        temp.append(x[:x1])
        temp.append(x[x3:x4])
        temp.append(x[x2:x3])
        temp.append(x[x1:x2])
        temp.append(x[x4:])
        return np.concatenate(temp,0)

    def OnePassage(self,x,mode,*args, **kwargs):
        x1,x2 = args[0],args[1]
        partial = x[x1:x2]
        if mode == 0: #逆序
            partial = self.Reverse(partial)
            x[x1:x2] = partial
            return x
        elif mode == 1: #噪声
            partial = self.AddNoise(partial)
            x[x1:x2] = partial
            return x
        elif mode == 2:#位移
            x3 = args[2]
            return self.Move(x,x1,x2,x3)
        else:
            return x

    def SelectOnePassageWithOnePoint(self,x):
        middle = statistics.median(x)
        sciles = np.where((x > middle - 0.1) & (x < middle + 0.1))[0]
        points = np.random.choice(sciles,3,replace=False)
        points =  np.sort(points)
        if np.random.random() > 0.5:
            return points
        else:
            return points[2],points[1],points[0]

    def SelectTwoPassage(self,x):
        middle = statistics.median(x)
        sciles = np.where((x > middle - 0.1) & (x < middle + 0.1))[0]
        points = np.random.choice(sciles,4)
        return np.sort(points)

    def WholePassage(self,x,mode,*args, **kwargs):
        if mode == 0: #逆序
            return self.Reverse(x)
        elif mode == 1: #噪声
            return self.AddNoise(x)
        else:
            return x

    def TwoPasssge(self,x,*args, **kwargs):
        return self.Change(x,*args)

    def __call__(self, x):
        result = list()
        x = np.array(x)
        num = 0
        #整段噪声
        while num < 50:
            line = self.WholePassage(x, mode=1)  # 噪声
            result.append(line.reshape(self.length))
            line = self.WholePassage(line, mode=0)  # 逆序
            result.append(line.reshape(self.length))
            num+=1
        #部分位移
        num = 0
        while num < 50:
            x1, x2, x3 = self.SelectOnePassageWithOnePoint(x)
            line = self.OnePassage(x, 2, x1, x2, x3)  # 位移
            if line.shape[0] == self.length:
                result.append(line.reshape(self.length))
                line = self.WholePassage(line, mode=0)  # 逆序
                result.append(line.reshape(self.length))
                num+=1
        result = np.array(result)
        # print(result.shape)
        #部分逆序
        num = 0
        while num < 50:
            while True :
                x1, x2, _ = self.SelectOnePassageWithOnePoint(x)
                if x2 - x1 > 0:
                    break
            line = self.OnePassage(x, 0, x1, x2)  # 部分逆序
            if line.shape[0] == self.length:
                result = np.concatenate([result,line.reshape(1,self.length)],0)
                line = self.Reverse(line)  # 逆序
                result = np.concatenate([result,line.reshape(1,self.length)],0)
                num+=1
        #两段交换
        num = 0
        while num < 50:
            while True:
                x1, x2, x3, x4 = self.SelectTwoPassage(x)
                if x2 - x1 > 0 and x3 > x2 and x4 > x3:
                    break
            line = self.TwoPasssge(x, x1, x2, x3, x4)  # 交换
            if line.shape[0] == self.length:
                result = np.concatenate([result, line.reshape(1, self.length)], 0)
                line = self.Reverse(line)  # 逆序
                result = np.concatenate([result, line.reshape(1, self.length)], 0)
                num += 1

        return result


def generate(line):
    tool = LineCreator()
    src = line[1:]
    result = tool(src)
    labels = np.ones((400,1)) * line[0]
    noise  = np.random.randn(400, 1)/1000
    labels += noise
    labels /= 20
    return np.concatenate([labels,result],1)


if __name__ == "__main__":
    base_path = "./dataset"
    bartons = np.load(op.join(base_path, "barton.npy"))
    result = list()
    result.extend(generate(line) for line in bartons)
    result = np.concatenate(result,0)
    print(result.shape)
    np.save(op.join(base_path, "dataset.npy"), result)

