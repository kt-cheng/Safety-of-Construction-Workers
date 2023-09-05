import pandas as pd

def covert(safe):
    
    safe.columns = [
        ['num', 'a1', 'a2', 'a3', 'b1', 'b2', 'b3', 'c1', 'c2', 'c3', 'd1', 'd2', 'd3', 'e1', 'e2', 'e3', 'f1', 'f2',
         'f3', 'g1', 'g2', 'g3', 'h1', 'h2', 'h3', 'i1', 'i2', 'i3', 'j1', 'j2', 'j3', 'k1', 'k2', 'k3', 'l1', 'l2',
         'l3', 'm1', 'm2', 'm3', 'n1', 'n2', 'n3', 'o1', 'o2', 'o3', 'p1', 'p2', 'p3', 'q1', 'q2', 'q3']]

    safe = safe.fillna(method='ffill')
    
    mask = safe[['a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'h3', 'i3', 'j3', 'k3', 'l3', 'm3', 'n3', 'o3', 'p3', 'q3']] <= 0.5
    data1 = safe[mask]
    
    data1['count'] = data1.sum(axis=1)
    data1['numm'] = (17 - data1[['a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'h3', 'i3', 'j3', 'k3', 'l3', 'm3', 'n3', 'o3', 'p3','q3']].isnull().sum(axis=1)).astype(int)
    data1['score'] = (data1[['count']].values / data1[['numm']].values)
    
    list_del = data1[data1[['numm']].values == [2]]
    
    mask2 = list_del[['a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'h3', 'i3', 'j3', 'k3', 'l3', 'm3', 'n3', 'o3', 'p3', 'q3']] > 0.5
    data2 = list_del[mask2]

    data3 = data2[['a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'h3', 'i3', 'j3', 'k3', 'l3', 'm3', 'n3', 'o3', 'p3', 'q3']]

    safe2 = safe.drop(data3.index, axis=0)
    
    res = pd.concat([safe2, data3], join='outer')
    res.sort_index(inplace=True)
    res2 = res.fillna(method='bfill')
    
    return res2