package null_value_remover;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

public class FilterNaNMapper extends Mapper<LongWritable, Text, LongWritable, Text> {
    private long validRowCount = 0; // Counter for valid rows

    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        // Skip the header row
        if (key.get() == 0 && value.toString().contains("id")) {
            context.write(key, value); // Pass the header to output
            return;
        }

        // Split the CSV line
        String[] columns = value.toString().split(",");

        // Check for "NaN" or empty values in the row
        boolean hasNaN = false;
        for (String column : columns) {
            if (column.equalsIgnoreCase("NaN") || column.isEmpty()) {
                hasNaN = true;
                break;
            }
        }

        // Emit only rows without "NaN" and count them
        if (!hasNaN) {
            context.write(key, value);
            validRowCount++; // Increment the counter for valid rows
        }
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        // Emit the total count of valid rows as a special key-value pair
        context.write(new LongWritable(-1), new Text("Valid rows count: " + validRowCount));
    }
}