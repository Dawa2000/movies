package column_selection;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

public class MovieReducer extends Reducer<LongWritable, Text, LongWritable, Text> {
    @Override
    protected void reduce(LongWritable key, Iterable<Text> values, Context context)
            throws IOException, InterruptedException {
        for (Text value : values) {
            context.write(key, value); // Pass the mapper output directly
        }
    }
}