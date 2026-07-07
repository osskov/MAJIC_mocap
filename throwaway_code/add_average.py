import re
import json
import collections
import base64
import struct

def decode_array(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and 'bdata' in data and 'dtype' in data:
        b = base64.b64decode(data['bdata'])
        dtype = data['dtype']
        if dtype == 'f8':
            return list(struct.unpack('<' + 'd' * (len(b) // 8), b))
        elif dtype == 'f4':
            return list(struct.unpack('<' + 'f' * (len(b) // 4), b))
    return data

filepath = 'data/mag_adapt_sensitivity_all.html'

with open(filepath, 'r') as f:
    content = f.read()

match = re.search(r'Plotly\.newPlot\(\s*[^,]+,\s*(\[.*?\])\s*,\s*(\{.*?\})\s*(?:,|\))', content, re.DOTALL)
if not match:
    print("Could not find Plotly traces in HTML.")
    exit(1)

traces_json = match.group(1)
traces = json.loads(traces_json)

# Remove any existing average traces (to avoid duplicating or averaging averages)
traces = [t for t in traces if 'Average' not in t.get('name', '')]

subplots = collections.defaultdict(list)
for t in traces:
    xaxis = t.get('xaxis', 'x')
    yaxis = t.get('yaxis', 'y')
    subplots[(xaxis, yaxis)].append(t)

new_traces = []

for (xaxis, yaxis), sub_traces in subplots.items():
    # We will track y values for three groups
    groups = {
        'All Subjects': collections.defaultdict(list),
        'Walking': collections.defaultdict(list),
        'Complex Tasks': collections.defaultdict(list),
    }
    
    for t in sub_traces:
        name = t.get('name', '')
        is_walking = 'walking' in name.lower()
        is_complex = 'complex' in name.lower()
        
        if 'x' in t and 'y' in t:
            x_arr = decode_array(t['x'])
            y_arr = decode_array(t['y'])
            for x, y in zip(x_arr, y_arr):
                if isinstance(y, (int, float)):
                    x_key = round(x, 5)
                    groups['All Subjects'][x_key].append(y)
                    if is_walking:
                        groups['Walking'][x_key].append(y)
                    if is_complex:
                        groups['Complex Tasks'][x_key].append(y)
    
    # Configuration for each average trace
    configs = [
        ('Average (All Subjects)', 'All Subjects', 'black', 'dash', 'star'),
        ('Average (Walking)', 'Walking', 'blue', 'dot', 'circle'),
        ('Average (Complex Tasks)', 'Complex Tasks', 'red', 'dot', 'triangle-up')
    ]
    
    for trace_name, group_key, color, dash, symbol in configs:
        x_to_ys = groups[group_key]
        if not x_to_ys:
            continue
            
        sorted_x = sorted(x_to_ys.keys())
        mean_y = [sum(x_to_ys[x]) / len(x_to_ys[x]) for x in sorted_x]
        
        avg_trace = {
            "type": "scatter",
            "mode": "lines+markers",
            "name": trace_name,
            "legendgroup": trace_name,
            "xaxis": xaxis,
            "yaxis": yaxis,
            "line": {"color": color, "width": 4, "dash": dash},
            "marker": {"symbol": symbol, "size": 10},
            "x": sorted_x,
            "y": mean_y
        }
        
        if xaxis == 'x' and yaxis == 'y':
            avg_trace['showlegend'] = True
        else:
            avg_trace['showlegend'] = False
            
        new_traces.append(avg_trace)

traces.extend(new_traces)
new_traces_json = json.dumps(traces)

start_idx = match.start(1)
end_idx = match.end(1)
new_content = content[:start_idx] + new_traces_json + content[end_idx:]

with open(filepath, 'w') as f:
    f.write(new_content)

print(f"Successfully added {len(new_traces)} Average lines to {filepath}.")
