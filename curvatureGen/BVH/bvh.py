import numpy as np
from typing import Tuple

from .ray import Ray
from .util import normalize, partition
from .bound import Bound3D, EmptyBound3D, unionBound, pointBound
from .primitive import Triangle


## BVH node definition
class BVHNode:
    """Base node implementation.
    """

    def __init__(self, bound:Bound3D, splitAxis:int):
        self.bound = bound
        self.splitAxis = splitAxis

class leafNode(BVHNode):
    """Leaf node implementation.
    """

    def __init__(self, idx: int, bound: Bound3D):
        super().__init__(bound, splitAxis=-1)

        self.triangleIdx = idx

class interiorNode(BVHNode):
    """Interior node implementation.
    """

    def __init__(self, splitAxis: int, left: BVHNode, right: BVHNode):
        bound = unionBound(left.bound, right.bound)
        super().__init__(bound, splitAxis)

        self.left = left
        self.right= right


## bucket definition
class BucketInfo:
    """Bucket info container.
    """

    def __init__(self, nBuckets: int):
        self.nBuckets = nBuckets

        self.count = np.zeros(self.nBuckets, dtype=int)
        self.bounds= [EmptyBound3D() for _ in range(self.nBuckets)]
    
    def push(self, idx: int, bound: Bound3D) -> None:
        """Push a bounding box to the given bucket.
        """

        self.count[idx] += 1
        self.bounds[idx] = unionBound(self.bounds[idx], bound)
    
    def SAH(self) -> np.ndarray:
        """Find cost for splitting buckets with SAH.

        Return:
            cost[nBuckets-1]: cost of splitting buckets
        """

        cost = np.zeros(self.nBuckets-1)
        totalSurfaceArea = self.boundsArea() + 1e-7

        for i in range(self.nBuckets-1):
            b0, b1 = EmptyBound3D(), EmptyBound3D()
            count0, count1 = 0, 0

            for j in range(i+1):
                b0 = unionBound(b0, self.bounds[j])
                count0 += self.count[j]
            
            for j in range(i+1, self.nBuckets):
                b1 = unionBound(b1, self.bounds[j])
                count1 += self.count[j]
            
            cost[i] = 0.125 + (count0*b0.surfaceArea() + count1*b1.surfaceArea()) / totalSurfaceArea

        return cost
    
    def boundsArea(self) -> float:
        """Calculate total surface area of all buckets.

        Return:
            A: surface area of the union of all boxes
        """

        box = EmptyBound3D()
        for bound in self.bounds:
            box = unionBound(box, bound)

        return box.surfaceArea()


## BVH definition
class BVH:
    """BVH acceleration.
    """

    def __init__(self, triangles:list[Triangle], nBuckets:int=12):
        self.triangles = triangles[:]
        self.nBuckets  = nBuckets

        ## build the BVH tree recursively
        orderedTriangles = []
        self.root = self._recursiveBuild(0, len(self.triangles), orderedTriangles)
        self.triangles = orderedTriangles
    
    def _recursiveBuild(self, start: int, end: int, orderedTriangles: list[Triangle]) -> BVHNode:
        """Build BVH recursively.

        Args:
            start: starting index
            end: ending index
            orderedTriangles: reordered triangles
        
        Return:
            node: BVH node
        """
        ## bound of all the triangles in the range [start, end)
        bounds = EmptyBound3D()
        for i in range(start, end):
            bounds = unionBound(bounds, self.triangles[i].bound())
        
        nTriangles = end - start
        ## create a leaf node if only one triangle is considered
        if nTriangles == 1:
            idx = len(orderedTriangles)
            orderedTriangles.append(self.triangles[start])
            return leafNode(idx, bounds)
        
        ## otherwise build the tree
        else:
            ## compute bound of primitive centroids, choose split dimension
            centroidBounds = EmptyBound3D()

            for i in range(start, end):
                centroid = self.triangles[i].centroid()
                centroidBounds = unionBound(centroidBounds, pointBound(centroid))
            
            dim = centroidBounds.maximumExtent()

            ## simply partition the triangles if all bounds have the same centroids
            if centroidBounds.pMax[dim] == centroidBounds.pMin[dim]:
                mid = (start + end) // 2
            else:
                ## partition triangles into two sets and build children
                if nTriangles == 2:
                    mid = (start + end) // 2
                    self.triangles[start: end] = sorted(self.triangles[start: end], key=lambda triangle: triangle.centroid()[dim])
                else:
                    buckets = BucketInfo(self.nBuckets)

                    ## distribute bounding boxes to buckets
                    for i in range(start, end):
                        b = self.nBuckets * centroidBounds.offset(self.triangles[i].centroid())[dim]
                        b = min(int(b), self.nBuckets-1)

                        buckets.push(b, self.triangles[i].bound())

                    ## find bucket to split at that minimizes SAH metric
                    cost = buckets.SAH()
                    minCostSplitBucket = np.argmin(cost)
                    # minCost = cost[minCostSplitBucket]

                    ## either create leaf or split primitives at selected SAH bucket
                    ## partition the triangles to two sets
                    def fun(triangle: Triangle):
                        b = self.nBuckets * centroidBounds.offset(triangle.centroid())[dim]
                        b = min(int(b), self.nBuckets-1)
                        return b <= minCostSplitBucket
                        
                    mid = partition(self.triangles, start, end, fun)
            
            node = interiorNode(dim, 
                                self._recursiveBuild(start, mid, orderedTriangles),
                                self._recursiveBuild(mid,   end, orderedTriangles))
        
        return node

    def intersect(self, ray: Ray):
        """Find the intersection of the ray from root.

        Args:
            ray: a ray instance
        
        Return:
            hit: whether the ray hits a triangle
            t: time of flight
            idx: triangle index
        """
        return self._intersect(ray, self.root)

    def _intersect(self, ray: Ray, node: BVHNode) -> Tuple[bool, float, int]:
        """Find intersection with a given ray and node.

        Args:
            ray: a ray instance
            node: current node
        
        Return:
            hit: whether the ray hits a triangle
            t: time of flight
            idx: triangle index
        """
        hit = node.bound.intersect(ray)
        if hit:
            if isinstance(node, leafNode):
                idx = node.triangleIdx
                return *self.triangles[idx].intersect(ray), idx
            else:
                hit1, t1, idx1 = self._intersect(ray, node.left)
                hit2, t2, idx2 = self._intersect(ray, node.right)

                if hit1 and hit2:
                    if t1 < t2:
                        return True, t1, idx1
                    else:
                        return True, t2, idx2
                elif hit1:
                    return True, t1, idx1
                elif hit2:
                    return True, t2, idx2
        
        return False, 0., -1