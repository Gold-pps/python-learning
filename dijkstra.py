# 迪杰斯特拉（Dijkstra）算法：求单源最短路径
# 前提：图中边的权重必须是非负数（有负权边时该算法不成立）

import heapq


def dijkstra(graph, start):
    """
    graph: 用邻接表表示的图，例如
           {
               "A": {"B": 5, "C": 1},
               "B": {"A": 5, "C": 2, "D": 1},
           }
           即 graph[节点] = {邻居节点: 边的权重}
    start: 起点节点

    返回 (dist, prev)：
      dist[节点] = 从 start 到该节点的最短距离
      prev[节点] = 最短路径上该节点的前一个节点（用于还原路径）
    """

    # 1. 初始化：所有节点的距离设为无穷大，起点设为 0
    dist = {node: float("inf") for node in graph}
    dist[start] = 0
    prev = {node: None for node in graph}

    # 优先队列中存放 (距离, 节点)，每次取出当前距离最小的节点
    pq = [(0, start)]

    while pq:
        d, node = heapq.heappop(pq)

        # 如果取出的距离比已知的最短距离大，
        # 说明该节点早已被更短路径处理过，直接跳过
        if d > dist[node]:
            continue

        # 2. 松弛操作：尝试通过 node 更新它所有邻居的最短距离
        for neighbor, weight in graph[node].items():
            new_dist = d + weight
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                prev[neighbor] = node
                heapq.heappush(pq, (new_dist, neighbor))

    return dist, prev


def shortest_path(prev, start, end):
    """根据 prev 表还原从 start 到 end 的具体路径（不含 start 时返回 None）"""
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()

    return path if path and path[0] == start else None


if __name__ == "__main__":
    # 示例：城市间公路及里程（双向道路）
    graph = {
        "北京": {"天津": 120, "石家庄": 300},
        "天津": {"北京": 120, "济南": 350},
        "石家庄": {"北京": 300, "郑州": 400},
        "济南": {"天津": 350, "南京": 600},
        "郑州": {"石家庄": 400, "南京": 700},
        "南京": {"济南": 600, "郑州": 700},
    }

    start, end = "北京", "南京"
    dist, prev = dijkstra(graph, start)

    path = shortest_path(prev, start, end)
    print(f"从 {start} 到 {end} 的最短路径：{' -> '.join(path)}")
    print(f"总距离：{dist[end]}")
