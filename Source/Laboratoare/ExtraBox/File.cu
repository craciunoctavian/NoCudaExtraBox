//#include "device_launch_parameters.h"
//#include "cuda_runtime.h"
//#include "Box.h"
//
//#define cudaCheckError() { \
//	cudaError_t e=cudaGetLastError(); \
//	if(e!=cudaSuccess) { \
//		printf("Cuda failure, %s",cudaGetErrorString(e)); \
//		exit(0); \
//	 }\
//}
//
//__global__ void check(int noOfCubes, box* boxes)
//{
//	int idxX = blockIdx.x * blockDim.x + threadIdx.x;
//	int idxY = blockIdx.y * blockDim.y + threadIdx.y;
//	if (idxX > noOfCubes || idxY > noOfCubes) {
//		return;
//	}
//
//	box b = boxes[idxX];
//	box c = boxes[idxY];
//	if (b.id == c.id) return;
//	// AABB collision
//	if ((c.x - 0.5 <= b.x + 0.5 && c.x + 0.5 >= b.x - 0.5) &&
//		(c.y - 0.5 <= b.y + 0.5 && c.y + 0.5 >= b.y - 0.5) &&
//		(c.z - 0.5 <= b.z + 0.5 && c.z + 0.5 >= b.z - 0.5)) {
//		if (c.collisions[b.id].id == -1) {
//			float new_m = boxes[c.id].m + boxes[b.id].m;
//			boxes[c.id].m = new_m;
//			boxes[b.id].m = new_m;
//			boxes[c.id].collisions[b.id] = b;
//			boxes[b.id].collisions[c.id] = c;
//			// if collision occurs the boxes drop
//			boxes[b.id].xVel = 0;
//			boxes[b.id].zVel = 0;
//			boxes[c.id].xVel = 0;
//			boxes[c.id].zVel = 0;
//			boxes[c.id].collisionsVector.insert(boxes[c.id].collisionsVector.begin(),
//				b.collisionsVector.begin(), b.collisionsVector.end());
//			boxes[b.id].collisionsVector.insert(boxes[b.id].collisionsVector.begin(),
//				c.collisionsVector.begin(), c.collisionsVector.end());
//			if (!b.moving) {
//				boxes[c.id].moving = false;
//				Animations::stopMoving(b);
//			}
//		}
//	}
//}
//
//bool bigCheck(box* h_boxes) {
//
//	//cudaMalloc((void**)&d_keys, numKeys * sizeof(int));
//	//cudaCheckError();
//	//cudaMalloc((void**)&d_values, numKeys * sizeof(int));
//	//cudaCheckError();
//
//	// Compute the parameters necessary to run the kernel: the number
//	// of blocks and the number of threads per block; also, deal with
//	// a possible partial final block
//	box* d_boxes;
//	cudaMemcpy(d_boxes, h_boxes, noOfCubes * sizeof(box), cudaMemcpyHostToDevice);
//	size_t blocks_no = noOfCubes;
//	check<<<blocks_no, noOfCubes>>>(noOfCubes, d_boxes);
//
//	// Wait for GPU to finish before accessing on host
//	cudaDeviceSynchronize();
//
//
//	//glbGpuAllocator->_cudaFree(d_keys);
//	//glbGpuAllocator->_cudaFree(_values);
//	return true;
//}
