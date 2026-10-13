"""The core raw-data invariant that caught the Census hierarchy bug.

A parent total must not be treated as a peer of its children. In the raw
Port-HS files, TOTAL FOR ALL PORTS and TOTAL FOR ALL COUNTRIES are hierarchy
nodes, not additional observations.
"""
def assert_hierarchy(parent_total, child_values, rtol=1e-8):
    child_total=sum(child_values)
    scale=max(abs(parent_total),1.0)
    assert abs(child_total-parent_total)/scale <= rtol, (parent_total,child_total)

# Correct aggregation pattern:
# product total:        all countries + all ports
# product-by-port:      all countries + specific port
# product-by-country:   specific country + all ports
# route edge:           specific country + specific port
