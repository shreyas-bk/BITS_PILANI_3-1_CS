`ifndef COMPARATOR_4BIT
`define COMPARATOR_4BIT
	module comparator_4bit(
		output agtb,
		output altb,
		output aeqb,
		input [3:0] a,
		input [3:0] b
	);
		
		wire a_i = {a[3],a};
		wire b_i = {b[3],b};
		
		assign agtb = $signed(a_i) > $signed(b_i) ;
		assign altb = $signed(a_i) < $signed(b_i) ;
		assign aeqb = ~agtb && ~altb;
		
	endmodule
`endif