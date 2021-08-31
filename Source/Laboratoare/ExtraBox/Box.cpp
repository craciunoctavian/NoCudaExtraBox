#include "Box.h"
#include <numeric>
#include <vector>
#include <algorithm>
#include <random>
#include <set>
#define M_PI 3.141592f

float velF = 0.1f;
float gravF = 0.005f;
std::map<int,box> boxes;
int noOfCubes = 100;
int id = 0;
float g = 9.8f;
float D = 5.5f;

// random float between 0 and 1
float randomFloat() {
	return float(rand()) / float((RAND_MAX));
}

void Animations::generateCube(float x, float y, float z) {
	float xVel = sin(randomFloat() * 2 * M_PI);
	float zVel = cos(randomFloat() * 2 * M_PI);
	boxes[id] = box{id, randomFloat() * 9 + 1, x, y, z, xVel, zVel, glm::vec3(randomFloat(), randomFloat(), randomFloat())};
	id++;
}

void Animations::initCubes() {
	srand((unsigned int)time(NULL));
	float x, y, z;
	for (int i = 0; i < noOfCubes; i++) {
		while (true) {
			bool collision = false;
			// generate random coordinates for cube
			x = float(rand()) / float((RAND_MAX)) * 22 - 11;
			y = float(rand()) / float((RAND_MAX)) * 22 - 11;
			z = float(rand()) / float((RAND_MAX)) * 22 - 11;
			// check to be in the extra box all the time
			if (abs(sqrt(x * x + z * z)) > 10) continue;

			// check collision at generation of cube
			for (int i = 0; i < noOfCubes; i++) {
				// we have collision between two boxes
				if (abs(x - boxes[i].x) < 1 && abs(y - boxes[i].y) < 1 && abs(z - boxes[i].z) < 1)
				{
					collision = true;
					break;
				}
			}
			if (!collision) break;
		}
		generateCube(x, y, z);
	}
}

void Animations::stopMoving(box b) {
	for (box c : b.collisionsVector) {
		boxes[c.id].moving = false;
	}

}

bool checkCollisionBigBox(box b) {

	float y = b.y - (-12);
	float zPos = b.z + 1 - (12);
	float zNeg = b.z - (-12);
	float xPos = b.x + 1 - (12);
	float xNeg = b.x - (-12);

	// collision with extraBox bottom
	if ((y < 0.5) && b.moving) {
		boxes[b.id].moving = false;
		Animations::stopMoving(b);
	}

	// collision with extraBox sides
	if (zPos > 0.5 || zNeg < 0.5 || xPos > 0.5 || xNeg < 0.5) {
		boxes[b.id].zVel = -boxes[b.id].zVel;
		boxes[b.id].xVel = -boxes[b.id].xVel;
	}
	return y < 0.5;
}


void Animations::moveCubes(float localTime) {

	for (int i = 0; i < noOfCubes; i++) {
		if (!boxes[i].moving) continue;
		checkCollision(boxes[i]);
	}

	for (int i = 0; i < noOfCubes; i++) {
		if (!boxes[i].moving) continue;
		if (!checkCollisionBigBox(boxes[i])) {
			boxes[i].y -= ((g - (float)(D / boxes[i].m)) / 2.0f) 
				* (float)(2.0f * localTime + 1.0f) * gravF;
			boxes[i].x += boxes[i].xVel * velF;
			boxes[i].z += boxes[i].zVel * velF;
		}
	}
}


void Animations::checkCollision(box c) {

	for (int i = 0; i < noOfCubes; i++) {
		box b = boxes[i];
		if (b.id == c.id) continue;
		// AABB collision
		if ((c.x - 0.5 <= b.x + 0.5 && c.x + 0.5 >= b.x - 0.5) &&
			(c.y - 0.5 <= b.y + 0.5 && c.y + 0.5 >= b.y - 0.5) &&
			(c.z - 0.5 <= b.z + 0.5 && c.z + 0.5 >= b.z - 0.5)) {
			if (c.collisions[b.id].id == -1) {
				float new_m = boxes[c.id].m + boxes[i].m;
				boxes[c.id].m = new_m;
				boxes[i].m = new_m;
				boxes[c.id].collisions[b.id] = b;
				boxes[i].collisions[c.id] = c;
				// if collision occurs the boxes drop
				boxes[i].xVel = 0;
				boxes[i].zVel = 0;
				boxes[c.id].xVel = 0;
				boxes[c.id].zVel = 0;
				boxes[c.id].collisionsVector.insert(boxes[c.id].collisionsVector.begin(), 
					b.collisionsVector.begin(), b.collisionsVector.end());
				boxes[i].collisionsVector.insert(boxes[b.id].collisionsVector.begin(), 
					c.collisionsVector.begin(), c.collisionsVector.end());
				if (!b.moving) {
					boxes[c.id].moving = false;
					stopMoving(b);
				}
			}
		}
	}
}