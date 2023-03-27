<h1>Image-based Rendering</h1>
<p>This is my 2013 C++ implementation of the Woodford et al 2008 paper "Global Stereo Reconstruction under Second Order Smoothness Priors" included here.</p>
<br/>
<p>Test suite not included.</p>
<br/>
<p>There are a few files sourced from other contributors:</p>
<li>IBR_Preprocess_GPU/arcball.h: KemboApi, The Turloc Toolkit, 1999-2003 Tatewake.com</li>
<li>IBR_Preprocess_GPU/arcball.c: KempoApi, The Turloc Toolkit, 1999-2003 Tatewake.com</li>
<li>IBR_Preprocess_GPU/decomposition.h: 2012 Philipp Wagner <bytefish[at]gmx[dot]de></li>
<li>IBR_Viewing/arcball.h: KempoApi, The Turloc Toolkit, 1999-2003 Tatewake.com</li>
<li>IBR_Viewing/arcball.c: KempoApi, The Turloc Toolkit, 1999-2003 Tatewake.com</li>
<br/>

<h2>IBR_Preprocess_GPU</h2>
<p>Computes stereo reconstruction from reference images.  It is GPU-accelerated.</p>

<h2>IBR_Viewing</h2>
<p>A viewer application to render pre-computed stereo reconstructions.  Synthesizes new views in real-time with mouse arcball controls.</p>
