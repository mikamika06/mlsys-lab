def greedy_pointwise_fuse(nodes):
    node_map = {n['id']: n for n in nodes}
    consumers = {n['id']: [] for n in nodes}
    for n in nodes:
        for inp in n['inputs']:
            if inp in consumers:
                consumers[inp].append(n['id'])

    groups = [[n['id']] for n in nodes]
    node_to_group = {n['id']: i for i, n in enumerate(nodes)}

    for n in nodes:
        curr_id = n['id']
        curr_group_idx = node_to_group[curr_id]

        for inp_id in list(n['inputs']):
            if inp_id not in node_map:
                continue
            inp_node = node_map[inp_id]
            inp_group_idx = node_to_group[inp_id]

            if curr_group_idx == inp_group_idx:
                continue

            if n['is_pointwise'] and inp_node['is_pointwise'] and n['shape'] == inp_node['shape']:
                if len(consumers[inp_id]) == 1:
                    target_group = groups[inp_group_idx]
                    source_group = groups[curr_group_idx]

                    target_group.extend(source_group)
                    for item in source_group:
                        node_to_group[item] = inp_group_idx
                    groups[curr_group_idx] = []

    return [g for g in groups if g]
