#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>

#include "nbody.cuh"


constexpr FLOAT THETA = 0.5f; // accuracy
constexpr FLOAT G = 1.0f;

struct Node
{
    FLOAT mass;
    FLOAT cmx, cmy, cmz; // center of Mass
    int particleIdx;     // index of particle if leaf, -1 if internal
    Node *children[8];

    Node()
    {
        mass = 0;
        cmx = 0;
        cmy = 0;
        cmz = 0;
        particleIdx = -1;
        for (int i = 0; i < 8; i++)
            children[i] = nullptr;
    }
};

// heap allocator
class NodePool
{
    std::vector<Node> pool;
    size_t current;

public:
    NodePool(size_t n)
    {
        pool.resize(n * 4); // we estimate the size of the pool in advance, avoiding heap allocation time penalty
        current = 0;
    }

    Node *alloc()
    {
        if (current >= pool.size())
            return nullptr;
        return &pool[current++];
    }
};

int getOctant(FLOAT x, FLOAT y, FLOAT z, FLOAT midX, FLOAT midY, FLOAT midZ)
{
    int octant = 0;
    if (x >= midX)
        octant |= 1;
    if (y >= midY)
        octant |= 2;
    if (z >= midZ)
        octant |= 4;
    return octant;
}

void insert(Node *node, int pIdx, Particle *particles,
            FLOAT x, FLOAT y, FLOAT z, FLOAT size, NodePool &pool)
{

    if (node->mass == 0 && node->particleIdx == -1 && node->children[0] == nullptr)
    {
        node->particleIdx = pIdx;
        node->mass = particles[pIdx].mass;
        node->cmx = particles[pIdx].x;
        node->cmy = particles[pIdx].y;
        node->cmz = particles[pIdx].z;
        return;
    }

    if (node->particleIdx != -1)
    {
        int existingP = node->particleIdx;
        node->particleIdx = -1;

        FLOAT midX = x + size * 0.5f;
        FLOAT midY = y + size * 0.5f;
        FLOAT midZ = z + size * 0.5f;

        int oct = getOctant(particles[existingP].x, particles[existingP].y, particles[existingP].z, midX, midY, midZ);

        if (node->children[oct] == nullptr)
            node->children[oct] = pool.alloc();

        FLOAT childSize = size * 0.5f;
        FLOAT childX = (oct & 1) ? midX : x;
        FLOAT childY = (oct & 2) ? midY : y;
        FLOAT childZ = (oct & 4) ? midZ : z;

        insert(node->children[oct], existingP, particles, childX, childY, childZ, childSize, pool);
    }

    FLOAT midX = x + size * 0.5f;
    FLOAT midY = y + size * 0.5f;
    FLOAT midZ = z + size * 0.5f;

    int oct = getOctant(particles[pIdx].x, particles[pIdx].y, particles[pIdx].z, midX, midY, midZ);

    if (node->children[oct] == nullptr)
        node->children[oct] = pool.alloc();

    FLOAT childSize = size * 0.5f;
    FLOAT childX = (oct & 1) ? midX : x;
    FLOAT childY = (oct & 2) ? midY : y;
    FLOAT childZ = (oct & 4) ? midZ : z;

    insert(node->children[oct], pIdx, particles, childX, childY, childZ, childSize, pool);
}

void computeMassDistribution(Node *node)
{
    if (!node)
        return;

    if (node->particleIdx == -1)
    {
        node->mass = 0;
        node->cmx = 0;
        node->cmy = 0;
        node->cmz = 0;

        for (int i = 0; i < 8; i++)
        {
            if (node->children[i])
            {
                computeMassDistribution(node->children[i]);
                FLOAT m = node->children[i]->mass;
                node->mass += m;
                node->cmx += node->children[i]->cmx * m;
                node->cmy += node->children[i]->cmy * m;
                node->cmz += node->children[i]->cmz * m;
            }
        }

        if (node->mass > 0)
        {
            node->cmx /= node->mass;
            node->cmy /= node->mass;
            node->cmz /= node->mass;
        }
    }
}

void computeForce(Node *node, FLOAT px, FLOAT py, FLOAT pz, FLOAT mass,
                  FLOAT &ax, FLOAT &ay, FLOAT &az,
                  FLOAT size, FLOAT softening)
{
    if (!node || node->mass == 0)
        return;

    FLOAT dx = node->cmx - px;
    FLOAT dy = node->cmy - py;
    FLOAT dz = node->cmz - pz;
    FLOAT distSq = dx * dx + dy * dy + dz * dz;
    FLOAT dist = sqrtf(distSq);

    if ((size / dist < THETA || node->particleIdx != -1) && dist > 0)
    {
        FLOAT f = (G * node->mass) / powf(distSq + softening * softening, 1.5f);
        ax += f * dx;
        ay += f * dy;
        az += f * dz;
    }
    else if (node->particleIdx == -1)
    {
        for (int i = 0; i < 8; i++)
        {
            computeForce(node->children[i], px, py, pz, mass, ax, ay, az, size * 0.5f, softening);
        }
    }
}

void nbody_barnes_hut_cpu(Particle *particles, int n, FLOAT dt, FLOAT softening)
{
    FLOAT minX = std::numeric_limits<FLOAT>::max(); // 32 or 64 bits max value
    FLOAT maxX = std::numeric_limits<FLOAT>::lowest();
    FLOAT minY = minX, maxY = maxX;
    FLOAT minZ = minX, maxZ = maxX;

    for (int i = 0; i < n; i++)
    {
        minX = std::min(minX, particles[i].x);
        maxX = std::max(maxX, particles[i].x);
        minY = std::min(minY, particles[i].y);
        maxY = std::max(maxY, particles[i].y);
        minZ = std::min(minZ, particles[i].z);
        maxZ = std::max(maxZ, particles[i].z);
    }

    // make the box cubic and centered
    FLOAT maxDim = std::max({maxX - minX, maxY - minY, maxZ - minZ});
    FLOAT centerX = (minX + maxX) / 2.0f;
    FLOAT centerY = (minY + maxY) / 2.0f;
    FLOAT centerZ = (minZ + maxZ) / 2.0f;

    // add small buffer
    FLOAT rootSize = maxDim * 1.01f;
    FLOAT rootX = centerX - rootSize / 2.0f;
    FLOAT rootY = centerY - rootSize / 2.0f;
    FLOAT rootZ = centerZ - rootSize / 2.0f;

    NodePool pool(n);
    Node *root = pool.alloc();

    for (int i = 0; i < n; i++)
    {
        insert(root, i, particles, rootX, rootY, rootZ, rootSize, pool);
    }

    computeMassDistribution(root);

    for (int i = 0; i < n; i++)
    {
        FLOAT ax = 0, ay = 0, az = 0;

        computeForce(root, particles[i].x, particles[i].y, particles[i].z,
                     particles[i].mass, ax, ay, az, rootSize, softening);

        particles[i].vx += ax * dt;
        particles[i].vy += ay * dt;
        particles[i].vz += az * dt;

        particles[i].x += particles[i].vx * dt;
        particles[i].y += particles[i].vy * dt;
        particles[i].z += particles[i].vz * dt;
    }
}
