import pandas as pd

def covert(dataset):
    
    dataset.columns = [
        ['num', 'a1', 'a2', 'a3', 'b1', 'b2', 'b3', 'c1', 'c2', 'c3', 'd1', 'd2', 'd3', 'e1', 'e2', 'e3', 'f1', 'f2',
         'f3', 'g1', 'g2', 'g3', 'h1', 'h2', 'h3', 'i1', 'i2', 'i3', 'j1', 'j2', 'j3', 'k1', 'k2', 'k3', 'l1', 'l2',
         'l3', 'm1', 'm2', 'm3', 'n1', 'n2', 'n3', 'o1', 'o2', 'o3', 'p1', 'p2', 'p3', 'q1', 'q2', 'q3']]

    dataset = dataset.fillna(method='ffill')
    
    confidence_mask_less = dataset[['a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'h3', 'i3', 'j3', 'k3', 'l3', 'm3', 'n3', 'o3', 'p3', 'q3']] <= 0.5
    filtered_data_1 = dataset[confidence_mask_less]
    
    filtered_data_1['count'] = filtered_data_1.sum(axis=1)
    filtered_data_1['num'] = (17 - filtered_data_1[['a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'h3', 'i3', 'j3', 'k3', 'l3', 'm3', 'n3', 'o3', 'p3','q3']].isnull().sum(axis=1)).astype(int)
    filtered_data_1['score'] = (filtered_data_1[['count']].values / filtered_data_1[['num']].values)
    
    equals_2_data = filtered_data_1[filtered_data_1[['num']].values == [2]]
    
    confidence_mask_greater = equals_2_data[['a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'h3', 'i3', 'j3', 'k3', 'l3', 'm3', 'n3', 'o3', 'p3', 'q3']] > 0.5
    filtered_data_2 = equals_2_data[confidence_mask_greater]

    extracted_data = filtered_data_2[['a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'h3', 'i3', 'j3', 'k3', 'l3', 'm3', 'n3', 'o3', 'p3', 'q3']]

    cleaned_data = dataset.drop(extracted_data.index, axis=0)
    
    result = pd.concat([cleaned_data, extracted_data], join='outer')
    result.sort_index(inplace=True)
    final_result = result.fillna(method='bfill')
    
    return final_result