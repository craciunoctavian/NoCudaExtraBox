#pragma once

#include <include/glm.h>
#include <map>
#include <vector>

struct box {
	int id = -1;
	float m = 1;
	float x = 0, y = 0, z = 0;
	float xVel = 0, zVel = 0;
	glm::vec3 color;
	bool moving = true;
	std::map<int, box> collisions;
	std::vector<box> collisionsVector;
};

extern std::map<int, box> boxes;
extern int noOfCubes;


namespace Animations {
	void generateCube(float x, float y, float z);
	void initCubes();
	void moveCubes(float localTime);
	void checkCollision(box b);
	void stopMoving(box b);
}