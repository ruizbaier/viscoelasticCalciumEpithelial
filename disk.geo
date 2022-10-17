SetFactory("OpenCASCADE");

R=1.43;


Circle(15) = {0, 0, 0, R, 0, 2*Pi};
Curve Loop(17)  = {15};
Plane Surface(22) = {17};


// IF WANTING 3D:
//Extrude {0,0,1} {Surface{22}; Layers{3};}


//Mesh.CharacteristicLengthFactor=0.09; //75
Mesh.CharacteristicLengthFactor = 0.05; //2D smaller disk of radius 1.43
Mesh.Algorithm=1;
Mesh.Format=1;
Mesh.ScalingFactor = 1.0;
Mesh.Smoothing = 20;

