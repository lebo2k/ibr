<h1>Image-based Rendering</h1>
This is my 2013 C++ implementation of the Woodford et al 2008 paper "Global Stereo Reconstruction under Second Order Smoothness Priors" included here.  There are a few files sourced from other contributors:
<li>IBR_Preprocess_GPU/arcball.h</li>
<li>IBR_Preprocess_GPU/arcball.c</li>
<li>IBR_Preprocess_GPU/decomposition.h</li>
<li>IBR_Viewing/arcball.h</li>
<li>IBR_Viewing/arcball.c</li>

<h2>IBR_Preprocess_GPU</h2>
Computes stereo reconstruction from reference images.  It is GPU-accelerated.

<h2>IBR_Viewing</h2>
A viewer application to render pre-computed stereo reconstructions.  Synthesizes new views in real-time with mouse arcball controls.
