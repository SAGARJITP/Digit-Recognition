#include "NeuralNet.h"

int NeuralNet::Run(void) {

	const int sizeInp = 112;//                   //default size of Test Image Box (112x112 pixels)
	PixelCol p(50, 275);//
	p.Initialize(sizeInp);//	
	int flag = 0;
	int flag_2 = 0;
	int flag_exit = 0;
	// Layer 1
	double **weights1 = ParseWeights("Weights/Weights_2.txt", 0, 16);
	double *bias1 = ParseBias("Weights/bias_2.txt", 0, 16);
	Layer1_Conv *conv_layer1 = new Layer1_Conv[1];

	Layer2_Relu *ReLU_layer2=new Layer2_Relu[1];
	layer3_MaxPool *MaxPool_layer3=new layer3_MaxPool[1];

	// Layer 4
	double ***weights4 = new double**[32];
	for (int filt_channel4 = 0; filt_channel4 < 32; filt_channel4++)
	{
		weights4[filt_channel4] = ParseWeights("Weights/Weights_5.txt", filt_channel4 * 16, (filt_channel4 + 1) * 16);
	}
	double *bias4 = ParseBias("Weights/bias_5.txt", 0, 32);
	Layer4_conv *conv_layer4 =new Layer4_conv[1];


	Layer5_Relu *ReLU_layer5=new Layer5_Relu[1];
	layer6_MaxPool *MaxPool_layer6 =new layer6_MaxPool[1];

	// Layer 7
	double ***weights7 = new double**[64];
	for (int filt_channel7 = 0; filt_channel7 < 64; filt_channel7++)
	{
		weights7[filt_channel7] = ParseWeights("Weights/Weights_8.txt", filt_channel7 * 32, (filt_channel7 + 1) * 32);
	}
	double *bias7 = ParseBias("Weights/bias_8.txt", 0, 64);


	Layer7_conv *conv_layer7 =new Layer7_conv[1];
	Layer8_Relu *ReLU_layer8 = new Layer8_Relu[1];

	// Specific to layer 9
	HugeWeights wt9(fopen("Weights/Weights_9.txt", "r")); //Weights
	double **weights9 = wt9.Run();

	Layer9_FullyConnected *FullyConnected_layer9 = new Layer9_FullyConnected[1];

	//specific to Layer_10
	Layer10_Softmax *Softmax_layer10 = new Layer10_Softmax[1];
	Layer10_Visualize *Visualize_layer10 = new Layer10_Visualize[1];
	double *layer_10_output;
	int n2 = 112;

	double** ImageOut = new double*[n2];
	for (int i = 0; i < n2; i++) {
		ImageOut[i] = new double[n2];
	}

	for (;;)
	{
		FsPollDevice();
		auto key = FsInkey();
		if (key != FSKEY_NULL)
		{
			break;
		}

		glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

		int lb, mb, rb, mx, my;
		FsGetMouseEvent(lb, mb, rb, mx, my);
		// Downsample
		
		p.Downsample(112,4, ImageOut);


		p.MouseEventLooping(p.bX, sizeInp, p.bY, lb, mb, rb, mx, my, flag, flag_2, flag_exit, ImageOut);
		if (flag_exit == 1) {
			//Cleanup
			delete[] conv_layer1, ReLU_layer2, MaxPool_layer3, conv_layer4, ReLU_layer5, MaxPool_layer6, conv_layer7, ReLU_layer8, FullyConnected_layer9, Softmax_layer10, Visualize_layer10, layer_10_output;
			delete[] weights1, bias1, weights4, bias4, weights7, bias7, weights9;
			return 1;
		}
		p.Draw(p.bX, p.bY, sizeInp);


		if (lb == 0 && flag == 1)
		{
			(*conv_layer1).conv(ImageOut, weights1, bias1);
			(*ReLU_layer2).ReLU((*conv_layer1).output_conv);
			(*MaxPool_layer3).maxP((*ReLU_layer2).output_ReLU);
			(*conv_layer4).conv((*MaxPool_layer3).output_maxP, weights4, bias4);
			(*ReLU_layer5).ReLU((*conv_layer4).output_conv);
			(*MaxPool_layer6).maxP((*ReLU_layer5).output_ReLU);
			(*conv_layer7).conv((*MaxPool_layer6).output_maxP, weights7, bias7);
			(*ReLU_layer8).ReLU((*conv_layer7).output_conv);
			(*FullyConnected_layer9).FullyConnected((*ReLU_layer8).output_ReLU, weights9);
			layer_10_output = (*Softmax_layer10).calculation((*FullyConnected_layer9).output_vector);
			//Visualize_layer10.fill(layer_10_output);
			(*Visualize_layer10).fill((*Softmax_layer10).soft_output);

			flag = 0;
			flag_2 = 1;
		}

		if (flag == 0 && flag_2 == 1)
		{
			(*conv_layer1).Show(6);
			(*MaxPool_layer3).show(6);
			(*conv_layer4).Show(9);
			(*MaxPool_layer6).show(9);
			(*conv_layer7).Show(12);
			(*Visualize_layer10).draw();
		}

		FsSwapBuffers();
		FsSleep(5);
	}
	glFlush();
	while (FSKEY_NULL == FsInkey())
	{
		return 0;
		FsPollDevice();
	}
}