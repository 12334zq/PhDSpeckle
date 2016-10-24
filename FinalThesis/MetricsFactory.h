/*Factory of correlation methods based on Metric class*/
#pragma once
#include "MetricSSD.h"
#include "MetricNSSD.h"
#include "MetricXC.h"
#include "MetricNXC.h"
#include "MetricZXC.h"
#include "MetricZNXC.h"
#include "MetricSAD.h"
#include "MetricMAD.h"

class MetricsFactory
{
public:
	/**
	Create and return pointer to specified Metric object 
	@param type				number of metric to use (see Metric::METRICS enum)
	@return					smart OpenCV pointer to initialized Metric object
	*/
   static Ptr<Metric> getMetric(int type){
	  
	   switch (type)
	   {
	   case Metric::SSD: return new MetricSSD();
	   case Metric::NSSD: return new MetricNSSD();
	   case Metric::XC: return new MetricXC();
	   case Metric::NXC: return new MetricNXC();
	   case Metric::ZXC: return new MetricZXC();
	   case Metric::ZNXC: return new MetricZNXC();
	   case Metric::SAD: return new MetricSAD();
	   case Metric::MAD: return new MetricMAD();
	   default: return new MetricSSD();
	   }

   }
};
