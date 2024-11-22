package null_value_remover;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class FilterNaNDriver {
    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: FilterNaNDriver <input path> <output path>");
            System.exit(-1);
        }

        // Configure the job
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Filter NaN Rows");
        job.setJarByClass(FilterNaNDriver.class);

        // Set Mapper and Reducer classes
        job.setMapperClass(FilterNaNMapper.class);
        job.setReducerClass(FilterNaNReducer.class);

        // Set output key and value types
        job.setOutputKeyClass(LongWritable.class);
        job.setOutputValueClass(Text.class);

        // Set input and output paths
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        // Run the job
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}